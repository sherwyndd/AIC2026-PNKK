import logging
import cv2
import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_CLIP_BATCH_SIZE, DEFAULT_KEYFRAME_STEP

_logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL_ID = "ViT-L-14-quickgelu"
DEFAULT_OPEN_CLIP_PRETRAINED = "dfn2b"


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


def filter_candidates_by_rel_diff(candidates, embeddings, threshold=0.4):
    """Filter candidates using L2-normalized relative difference (Vortex paper)."""
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


class CLIPEmbedder:
    """CLIP Feature Extractor supporting OpenCLIP (ViT-L-14-quickgelu dfn2b) and HuggingFace transformers."""

    def __init__(
        self,
        device,
        model_id=DEFAULT_CLIP_MODEL_ID,
        pretrained=DEFAULT_OPEN_CLIP_PRETRAINED,
        batch_size=DEFAULT_CLIP_BATCH_SIZE,
    ):
        self.device = device
        self.batch_size = batch_size
        self.use_open_clip = False
        self._embed_dim = 768

        try:
            import open_clip

            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_id if "/" not in model_id else "ViT-L-14-quickgelu",
                pretrained=pretrained if pretrained else "dfn2b",
                device=device,
            )
            self.model.eval()
            self.use_open_clip = True
            _logger.info(f"Loaded OpenCLIP model '{model_id}' ({pretrained}) on {device}")
        except Exception as exc:
            _logger.warning(f"OpenCLIP load failed ({exc}). Falling back to HuggingFace transformers...")
            from transformers import CLIPModel, CLIPProcessor

            hf_model_id = "openai/clip-vit-large-patch14" if "/" not in model_id else model_id
            self.processor = CLIPProcessor.from_pretrained(hf_model_id)
            self.model = CLIPModel.from_pretrained(hf_model_id).to(device)
            self.model.eval()
            self.use_open_clip = False

    def encode_images(self, frames_bgr):
        """Encode list of BGR frames to normalized float32 feature array of shape (N, 768)."""
        if not frames_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        chunks = []
        curr_batch_size = max(1, self.batch_size)
        idx = 0

        while idx < len(frames_bgr):
            batch = frames_bgr[idx : idx + curr_batch_size]
            try:
                if self.use_open_clip:
                    tensors = []
                    for frame in batch:
                        if frame is None or frame.size == 0:
                            blank = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                            tensors.append(self.preprocess(blank))
                        else:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            tensors.append(self.preprocess(Image.fromarray(rgb)))
                    input_batch = torch.stack(tensors).to(self.device)

                    with torch.no_grad():
                        feats = self.model.encode_image(input_batch)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    arr = feats.cpu().numpy().astype(np.float32)
                else:
                    rgb_images = [
                        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if f is not None and f.size > 0
                        else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                        for f in batch
                    ]
                    inputs = self.processor(images=rgb_images, return_tensors="pt", padding=True)
                    pixel_values = inputs["pixel_values"].to(self.device)
                    with torch.no_grad():
                        feats = self.model.get_image_features(pixel_values=pixel_values)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                    arr = feats.cpu().numpy().astype(np.float32)

                chunks.append(arr)
                idx += len(batch)

            except Exception as exc:
                if curr_batch_size > 1 and ("out of memory" in str(exc).lower() or "oom" in str(exc).lower()):
                    curr_batch_size = max(1, curr_batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    _logger.error(f"CLIP encode error: {exc}. Using zero vectors for batch.")
                    chunks.append(np.zeros((len(batch), self._embed_dim), dtype=np.float32))
                    idx += len(batch)

        return np.concatenate(chunks, axis=0)
