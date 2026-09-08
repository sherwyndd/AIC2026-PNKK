"""BGE-Reranker-v2-m3 cross-encoder module with smart metadata fallback and Min-Max normalization."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_reranker_model = None
_reranker_device = None


def load_bge_reranker(device: str = "cuda:0"):
    """Load BAAI/bge-reranker-v2-m3 cross-encoder model onto device using float16."""
    global _reranker_model, _reranker_device
    _reranker_device = device
    logger.info("Loading BGE-Reranker-v2-m3 on %s ...", device)
    try:
        from sentence_transformers import CrossEncoder

        automodel_args = {}
        if "cuda" in device and torch.cuda.is_available():
            automodel_args["torch_dtype"] = torch.float16

        _reranker_model = CrossEncoder(
            MODEL_NAME,
            max_length=512,
            device=device,
            automodel_args=automodel_args,
        )
        logger.info("BGE-Reranker-v2-m3 loaded successfully on %s (dtype=float16).", device)
    except Exception as exc:
        logger.exception("Failed to load BGE-Reranker-v2-m3 (%s): %s", MODEL_NAME, exc)
        _reranker_model = None
    return _reranker_model


def rerank_candidates(
    query: str,
    candidates: List[Dict],
    top_k: int,
    rerank_top_n: int = 50,
) -> List[Dict]:
    """
    Rerank candidates using BGE-Reranker-v2-m3 ONLY when valid text metadata exists,
    preserving exact visual search precision for pure visual hits.

    Args:
        query: User query string (e.g. Vietnamese / English text).
        candidates: Candidate result dicts from first stage (RRF / Vector / ES).
        top_k: Desired output length.
        rerank_top_n: Number of candidate items to rerank.

    Returns:
        List of reranked candidates, length <= top_k.
    """
    if not query or not candidates:
        return candidates[:top_k]

    if _reranker_model is None:
        logger.warning("BGE Reranker model is not loaded, returning original rank.")
        return candidates[:top_k]

    pool = candidates[:rerank_top_n]
    rest = candidates[rerank_top_n:]

    # Separate items that actually have text metadata (OCR, ASR, Object text)
    valid_pairs = []
    valid_indices = []

    for idx, item in enumerate(pool):
        meta_parts = []
        if item.get("ocr_text"):
            meta_parts.append(f"OCR: {item['ocr_text']}")
        if item.get("asr_text"):
            meta_parts.append(f"ASR: {item['asr_text']}")
        if item.get("object_text"):
            meta_parts.append(f"Vật thể: {item['object_text']}")

        if meta_parts:
            doc_text = " | ".join(meta_parts)
            valid_pairs.append((query, doc_text))
            valid_indices.append(idx)

    # 🛑 If NO candidate has text metadata (pure visual search from SigLIP2),
    # preserve 100% sharp visual ranking directly!
    if not valid_pairs:
        logger.info("Pure visual search: Preserving 100% sharp SigLIP2 visual rank.")
        return candidates[:top_k]

    try:
        raw_scores = _reranker_model.predict(valid_pairs, batch_size=32, show_progress_bar=False)

        # Normalize logit scores to (0, 1] using sigmoid
        sigmoid_scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))

        # Min-Max normalize original scores of the candidate pool to [0, 1] for fair blending
        orig_scores = [float(item.get("score", 0.0)) for item in pool]
        min_s, max_s = min(orig_scores), max(orig_scores)
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        for i, idx in enumerate(valid_indices):
            item = pool[idx]
            ce_score = float(sigmoid_scores[i])
            norm_orig_score = (float(item.get("score", 0.0)) - min_s) / range_s

            # Blend 60% normalized original rank score + 40% Cross-Encoder text score
            item["score"] = round(0.6 * norm_orig_score + 0.4 * ce_score, 6)
            if "matched_sources" in item:
                if "bge_reranker" not in item["matched_sources"]:
                    item["matched_sources"].append("bge_reranker")

        pool_sorted = sorted(pool, key=lambda x: x.get("score", 0.0), reverse=True)
        return (pool_sorted + rest)[:top_k]
    except Exception as exc:
        logger.exception("Error during BGE reranking, falling back to original ranks: %s", exc)
        return candidates[:top_k]
