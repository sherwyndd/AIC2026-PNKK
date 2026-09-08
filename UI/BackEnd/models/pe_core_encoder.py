"""PE-Core-L encoder — loads model once via open_clip, exposes encode_text_pecore().

Aligned with nhan/data-AIC2025/src/embedding/encode_pe_core.py:
- model_id: hf-hub:timm/PE-Core-L-14-336
- float16 on CUDA, float32 on CPU
- normalize=True when encoding
- inference_mode() instead of no_grad()
- Embedding dim: 1024
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "hf-hub:timm/PE-Core-L-14-336"
TOKENIZER_LANG = "en"

_pecore_model = None
_pecore_preprocess = None
_pecore_tokenizer = None
_pecore_device = None
_pecore_use_half = False


def load_pecore(device: str = "cuda:0"):
    """Load PE-Core-L via open_clip onto *device*.

    Returns (model, preprocess, tokenizer).
    Uses float16 on CUDA for efficiency.
    """
    global _pecore_model, _pecore_preprocess, _pecore_tokenizer, _pecore_device, _pecore_use_half

    import open_clip

    logger.info("Loading PE-Core-L on %s …", device)
    _pecore_device = torch.device(device)
    _pecore_use_half = _pecore_device.type == "cuda"

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
    tokenizer = open_clip.get_tokenizer(MODEL_ID)

    model = model.to(_pecore_device).eval()
    if _pecore_use_half:
        model = model.half()

    _pecore_model = model
    _pecore_preprocess = preprocess
    _pecore_tokenizer = tokenizer

    logger.info(
        "PE-Core-L loaded on %s (dtype=%s).",
        _pecore_device,
        "float16" if _pecore_use_half else "float32",
    )
    return _pecore_model, _pecore_preprocess, _pecore_tokenizer


def encode_text_pecore(text: str) -> np.ndarray:
    """
    Encode a single text query with PE-Core-L.
    Returns a float32 numpy array of shape (1024,).
    """
    if _pecore_model is None or _pecore_tokenizer is None:
        raise RuntimeError("PE-Core model is not loaded. Call load_pecore() first.")

    tokens = _pecore_tokenizer([text]).to(_pecore_device)

    with torch.inference_mode():
        if _pecore_use_half:
            with torch.autocast("cuda", dtype=torch.float16):
                feats = _pecore_model.encode_text(tokens, normalize=True)
        else:
            feats = _pecore_model.encode_text(tokens, normalize=True)

    emb = feats.float().cpu().numpy()[0]
    return emb.astype(np.float32)
