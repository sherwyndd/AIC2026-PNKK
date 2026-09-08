"""Reciprocal Rank Fusion (RRF) implementation."""

from typing import Dict, List


def rrf_fuse(
    results_by_model: Dict[str, list],
    k: int,
    top_k: int,
) -> List[dict]:
    """
    Merge ranked result lists from multiple models using RRF.

    Formula:
        score(image) = Σ_model  1 / (k + rank_in_model)   (rank is 1-indexed)

    Args:
        results_by_model: dict mapping model name → list of ScoredPoint
                          (from qdrant_client, each item has .payload).
        k:                RRF k hyper-parameter (typically 60).
        top_k:            How many fused results to return.

    Returns:
        List of dicts sorted by descending RRF score, length ≤ top_k.
        Each dict: {"score": float, **payload fields}
    """
    scores: Dict[str, float] = {}
    meta: Dict[str, dict] = {}

    for _model_name, hits in results_by_model.items():
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload
            key: str = payload["keyframe_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in meta:
                meta[key] = payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    non_empty_models = sum(1 for hits in results_by_model.values() if hits)
    max_score = non_empty_models * (1.0 / (k + 1)) if non_empty_models > 0 else 1.0

    return [
        {"score": round((score / max_score) if max_score > 0 else 0.0, 6), **meta[key]}
        for key, score in ranked
    ]
