import logging

import cv2
import numpy as np
import torch

from .config import DEFAULT_CLIP_BATCH_SIZE, DEFAULT_CLIP_MODEL_ID, DEFAULT_KEYFRAME_STEP

_logger = logging.getLogger(__name__)


def get_target_frames(start_frame, end_frame, step=DEFAULT_KEYFRAME_STEP):
    return list(range(start_frame, end_frame + 1, step))


def read_frame_with_fallback(cap, target_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if ret:
        return target_frame, frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame - 1))
    ret, frame = cap.read()
    if ret:
        return max(0, target_frame - 1), frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame + 1)
    ret, frame = cap.read()
    if ret:
        return target_frame + 1, frame

    return None, None


def collect_shot_candidates(cap, start_frame, end_frame, sample_step):
    candidates = []
    for target_frame in get_target_frames(start_frame, end_frame, step=sample_step):
        actual_frame_idx, frame = read_frame_with_fallback(cap, target_frame)
        if frame is not None:
            candidates.append((actual_frame_idx, frame))
    return candidates


def filter_candidates_by_rel_diff(candidates, embeddings, threshold):
    if not candidates:
        return []

    selected = [candidates[0]]
    e_prev = embeddings[0].astype(np.float64)
    norm_prev = np.linalg.norm(e_prev)
    if norm_prev == 0:
        norm_prev = 1e-8

    for idx in range(1, len(candidates)):
        e_current = embeddings[idx].astype(np.float64)
        rel_diff = np.linalg.norm(e_current - e_prev) / norm_prev
        if rel_diff > threshold:
            selected.append(candidates[idx])
            e_prev = e_current
            norm_prev = np.linalg.norm(e_prev)
            if norm_prev == 0:
                norm_prev = 1e-8

    return selected


def _extract_tensor_from_features(features):
    """Extract torch.Tensor from all known CLIPModel.get_image_features() return formats."""
    if isinstance(features, torch.Tensor):
        tensor = features
    elif isinstance(features, (tuple, list)):
        # Some transformers versions return (pooled_output,) or (last_hidden, pooled, ...)
        first = features[0]
        if not isinstance(first, torch.Tensor):
            raise RuntimeError(
                f"CLIP returned a {type(features).__name__} whose first element is "
                f"{type(first)} (not a Tensor)."
            )
        tensor = first
    elif hasattr(features, "image_embeds"):
        tensor = features.image_embeds
    elif isinstance(features, dict) and "image_embeds" in features:
        tensor = features["image_embeds"]
    elif hasattr(features, "pooler_output"):
        tensor = features.pooler_output
    else:
        raise RuntimeError(
            f"Unexpected CLIP image feature output type: {type(features)}. "
            "Expected a Tensor, tuple/list, object/dict with 'image_embeds', or object with 'pooler_output'."
        )

    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(
            f"Extracted CLIP feature is {type(tensor)}, not a torch.Tensor."
        )
    return tensor


class CLIPEmbedder:
    def __init__(self, device, model_id=DEFAULT_CLIP_MODEL_ID, batch_size=DEFAULT_CLIP_BATCH_SIZE):
        try:
            from transformers import CLIPModel, CLIPProcessor
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "transformers and pillow are required for CLIP filtering. "
                "Install with: pip install transformers accelerate pillow"
            ) from exc

        self.device = device
        self.batch_size = batch_size
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.Image = Image

        # Infer embedding dim from model config — avoids hardcoding 768 or 512
        try:
            self._embed_dim = self.model.config.projection_dim
        except AttributeError:
            self._embed_dim = 768  # fallback for clip-vit-large-patch14

    def encode_images(self, frames_bgr):
        """
        Encode a list of BGR frames to a float32 array of shape (N, embed_dim).
        Bad frames are replaced with zero vectors so the pipeline never stops mid-run.
        Includes dynamic batch size reduction on CUDA OOM.
        """
        if not frames_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        chunks = []
        curr_batch_size = max(1, self.batch_size)
        idx = 0

        while idx < len(frames_bgr):
            batch = frames_bgr[idx : idx + curr_batch_size]
            try:
                rgb_images = []
                for frame in batch:
                    if frame is None or frame.size == 0:
                        _logger.warning("Skipping empty/None frame in CLIP batch — using blank.")
                        blank = np.zeros((224, 224, 3), dtype=np.uint8)
                        rgb_images.append(self.Image.fromarray(blank))
                    else:
                        rgb_images.append(
                            self.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        )

                inputs = self.processor(images=rgb_images, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(self.device)

                with torch.no_grad():
                    features = self.model.get_image_features(pixel_values=pixel_values)

                tensor = _extract_tensor_from_features(features)
                arr = tensor.detach().cpu().float().numpy()
                chunks.append(arr)
                idx += len(batch)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                err_msg = str(exc).lower()
                if ("out of memory" in err_msg or "oom" in err_msg) and curr_batch_size > 1:
                    new_batch_size = max(1, curr_batch_size // 2)
                    _logger.warning(
                        "CUDA OOM encountered with batch_size=%d. Reducing batch_size to %d and retrying...",
                        curr_batch_size, new_batch_size
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    curr_batch_size = new_batch_size
                    continue
                else:
                    _logger.error(
                        "CLIP encode failed for batch [%d:%d]: %s — replacing with zero vectors.",
                        idx, idx + len(batch), exc,
                    )
                    chunks.append(np.zeros((len(batch), self._embed_dim), dtype=np.float32))
                    idx += len(batch)

        return np.concatenate(chunks, axis=0)

