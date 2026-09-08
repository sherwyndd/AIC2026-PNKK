"""SigLIP2 encoder — loads model once, exposes encode_text_siglip().

Aligned with nhan/data-AIC2025/src/embedding/siglip2.py:
- Model: google/siglip2-so400m-patch14-384 (1152-dim, image 384x384)
- float16 dtype on CUDA, float32 on CPU
- trust_remote_code=False
- inference_mode() instead of no_grad()
- Full fallback extraction kept for safety

[UPGRADE 31/08] so400m-patch14-384 thay cho base-patch16-224:
  - dim: 768 -> 1152
  - image size: 224 -> 384 (patch14 thay vi patch16)
  - Phai khop voi siglip2_final/ embeddings cua Nhan
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# [UPGRADE 31/08] so400m-patch14-384 (1152-dim, 384x384) — khop voi
# nhan/data-AIC2025/outputs/siglip2_final/ embeddings.
# Model cu: google/siglip2-base-patch16-224 (768-dim, 224x224)
MODEL_NAME = "google/siglip2-so400m-patch14-384"

_siglip_model = None
_siglip_processor = None
_siglip_device = None
_siglip_use_half = False


def load_siglip(device: str = "cuda:0"):
    """Load google/siglip2-so400m-patch14-384 via transformers onto *device*.

    Returns (processor, model).
    Uses float16 on CUDA (matching the offline embedding pipeline).
    """
    global _siglip_model, _siglip_processor, _siglip_device, _siglip_use_half

    from transformers import AutoModel, AutoProcessor

    logger.info("Loading SigLIP2 on %s …", device)
    _siglip_device = torch.device(device)
    _siglip_use_half = _siglip_device.type == "cuda"

    _siglip_processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _siglip_model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if _siglip_use_half else torch.float32,
        trust_remote_code=False,
    )
    _siglip_model = _siglip_model.to(_siglip_device).eval()

    logger.info(
        "SigLIP2 loaded on %s (dtype=%s).",
        _siglip_device,
        "float16" if _siglip_use_half else "float32",
    )
    return _siglip_processor, _siglip_model


def _extract_features(outputs, modality: str) -> torch.Tensor:
    """
    Unified feature extraction — SAME LOGIC for text and image.
    Priority order (most-aligned first):
      1. Direct tensor output (e.g., modern get_*_features())
      2. `{text,image}_embeds`  (post-projection head — CONTRASTIVE-ALIGNED)
      3. `pooler_output`         (pre-projection head, usually <[BOS]> token)
      4. `last_hidden_state[:,0,:]` (raw <[BOS]> token)
    """
    if isinstance(outputs, torch.Tensor):
        return outputs

    embeds_key = f"{modality}_embeds"
    if hasattr(outputs, embeds_key) and getattr(outputs, embeds_key) is not None:
        return getattr(outputs, embeds_key)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state[:, 0, :]

    return outputs[0]


def encode_text_siglip(text: str) -> np.ndarray:
    """Encode a single text query -> L2-normalized float32 (1152,) embedding.

    [UPGRADE 31/08] so400m-patch14-384: dim=1152, dummy pixel 384x384.

    CRITICAL: HuggingFace SiglipModel.forward() *requires* BOTH text+image inputs
        (otherwise vision_model crashes on pixel_values=None).  `get_text_features()`
        only runs text_model and does NOT apply the text-model projection head,
        which breaks cross-modal alignment.  To guarantee we obtain the same
        `text_embeds` used during contrastive training, we feed a DUMMY zero
        pixel_values tensor (N, 3, 384, 384) alongside the real text tokens,
        then return outputs['text_embeds'] (already L2-normalised by the model).
    """
    if _siglip_model is None or _siglip_processor is None:
        raise RuntimeError("SigLIP model is not loaded. Call load_siglip() first.")

    # SigLIP2 tokenizer returns ONLY input_ids (no attention_mask), so we
    # use the text-specific branch of the processor.
    text_inputs = _siglip_processor(
        text=[text],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(_siglip_device)

    # Build dummy pixel_values so SiglipModel.forward can run both branches.
    # [UPGRADE 31/08] so400m-patch14-384 xu ly anh 384x384, khong phai 224x224.
    # Shape convention from SiglipImageProcessor: (N, 3, 384, 384).
    dtype = torch.float16 if _siglip_use_half else torch.float32
    B = input_ids.shape[0]
    dummy_pixel = torch.zeros(B, 3, 384, 384, dtype=dtype, device=_siglip_device)

    with torch.inference_mode():
        outputs = _siglip_model(
            input_ids=input_ids,
            pixel_values=dummy_pixel,
            return_loss=False,
        )

    # `text_embeds` is L2-normalised inside SigLipModel forward (contrastive space)
    # [UPGRADE 31/08] shape: (1152,) voi so400m-patch14-384
    text_emb = outputs["text_embeds"][0].cpu().float().numpy()
    return text_emb.astype(np.float32)


def encode_images_siglip(images: list) -> np.ndarray:
    """Encode a batch of PIL images -> L2-normalised float32 (N, 1152) embeddings.

    [UPGRADE 31/08] so400m-patch14-384: dim=1152, image size=384x384.
    Uses the SAME canonical SiglipModel.forward path with DUMMY text tokens so the
    returned `image_embeds` runs through vision_model.head (attention pooling
    projection head) which is bit-exact to contrastive space used during training.
    """
    if _siglip_model is None or _siglip_processor is None:
        raise RuntimeError("SigLIP model is not loaded. Call load_siglip() first.")

    img_inputs = _siglip_processor(images=images, return_tensors="pt")
    img_dtype = torch.float16 if _siglip_use_half else torch.float32
    pixel_values = img_inputs["pixel_values"].to(device=_siglip_device, dtype=img_dtype)

    # Dummy text input_ids (shape [B_image, 64] filled with pad/eos id 1) – SigLIP2 padding_id=1).
    B = pixel_values.shape[0]
    dtype = torch.long
    dummy_ids = torch.full((B, 64), fill_value=1, dtype=dtype, device=_siglip_device)

    with torch.inference_mode():
        outputs = _siglip_model(
            input_ids=dummy_ids,
            pixel_values=pixel_values,
            return_loss=False,
        )

    # `image_embeds` (B, 1152) already L2-normalised by SigLip forward.
    # [UPGRADE 31/08] dim=1152 voi so400m-patch14-384
    image_embs = outputs["image_embeds"].cpu().float().numpy()
    return image_embs.astype(np.float32)
