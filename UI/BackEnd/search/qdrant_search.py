"""Qdrant query helpers."""

import logging
import time
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    SearchParams,
)

logger = logging.getLogger(__name__)

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333

COLLECTION_PECORE = "image_pecore"
# [UPGRADE] SigLIP collection toggle:
#   "image_siglip"       -> ĐANG DÙNG: FLOAT32 full, HNSW m=32/ef_construct=512
#                            (recall tối đa, không quantize — RAM dư dả nên dùng bản này.
#                            LƯU Ý: script build mới đã XOÁ và TẠO LẠI collection này với
#                            config nâng cấp, khác với bản "image_siglip" gốc cũ trước đây)
#   "image_siglip_sq8"   -> Scalar Quantization INT8 (nhanh hơn, recall thấp hơn 1 chút)
# Nếu cần rollback về SQ8 (ưu tiên tốc độ hơn recall), đổi lại "image_siglip_sq8"
# và bật lại QuantizationSearchParams trong _build_search_params() (xem git history).
COLLECTION_SIGLIP = "image_siglip"

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return a shared QdrantClient instance (created on first call)."""
    global _client
    if _client is None:
        _client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT, timeout=120)
        logger.info("Qdrant client connected to %s:%s (timeout=120s)", QDRANT_HOST, QDRANT_PORT)
    return _client


def is_qdrant_healthy() -> bool:
    """Ping Qdrant — returns True if responsive."""
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:
        return False


def _build_filter(video_list: Optional[List[str]]) -> Optional[Filter]:
    """Build a Qdrant Filter for video_id if video_list is non-empty, else None."""
    if not video_list:
        return None
    return Filter(
        must=[FieldCondition(key="video_id", match=MatchAny(any=video_list))]
    )


def query_collection(
    embedding: np.ndarray,
    collection_name: str,
    limit: int,
    video_list: Optional[List[str]] = None,
) -> list:
    """
    Query a Qdrant collection with the given embedding vector.
    Features granular stage timing:
    - vector conversion
    - client call (HTTP send + server processing + receive)
    - point extraction / response parsing

    [UPGRADE] Collection mặc định (image_siglip_fp32_hq) không còn dùng
    quantization -> không cần QuantizationSearchParams/rescore nữa, vì
    KHÔNG có bước nén nào để rescore bù lại. search_params giờ chỉ còn
    hnsw_ef + indexed_only, đơn giản hơn hẳn so với bản SQ8 trước đây.

    Nếu collection_name trỏ ngược về "image_siglip_sq8" (rollback), cần
    thêm lại QuantizationSearchParams(rescore=True, oversampling=2.0)
    thủ công, nếu không recall sẽ bị giảm do dùng thẳng vector nén mà
    không rescore (sai lệch âm thầm, không có exception báo hiệu).
    """
    t_stage_start = time.perf_counter()
    client = get_qdrant_client()
    query_filter = _build_filter(video_list)
    query_vector = embedding.flatten().tolist()
    t_prep = time.perf_counter() - t_stage_start

    logger.debug(
        "[QDRANT_TIMING] Prep vector dim=%d took %.4fs for collection=%s",
        len(query_vector),
        t_prep,
        collection_name,
    )

    try:
        try:
            safe_limit = int(limit)
        except Exception:
            safe_limit = int(float(limit))

        search_params = _build_search_params()

        # 1. Before client.query_points()
        t_before_call = time.perf_counter()
        logger.info(
            "[QDRANT_TIMING] -> Entering client.query_points(collection=%s, limit=%d, "
            "hnsw_ef=256, indexed_only=True, no_quantization=True)",
            collection_name,
            safe_limit,
        )

        # 2. Execute query_points
        # [UPGRADE] hnsw_ef=256 (tăng từ 128): không còn quantization nên có
        # thêm "budget" tốc độ để tăng ef, đổi lấy recall cao hơn.
        # [GIỮ NGUYÊN] indexed_only=True: free lunch với dữ liệu fixed
        # (không insert liên tục). Nếu sau này insert thêm video mới liên
        # tục, cân nhắc đổi False để không bỏ sót điểm chưa kịp index.
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=safe_limit,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=["video_id", "keyframe_id", "frame_idx", "image_path", "shot_id"],
            timeout=120,
        )
        t_after_call = time.perf_counter()
        http_and_server_time = t_after_call - t_before_call

        # 3. Response parsing
        t_parse_start = time.perf_counter()
        if hasattr(result, 'points'):
            points = result.points
        elif isinstance(result, list):
            points = result
        else:
            points = []
        t_parse = time.perf_counter() - t_parse_start
        total_qdrant_time = time.perf_counter() - t_stage_start

        logger.info(
            "[QDRANT_TIMING] <- client.query_points() completed: HTTP+Server=%.4fs | parse=%.4fs | total=%.4fs (returned %d points)",
            http_and_server_time,
            t_parse,
            total_qdrant_time,
            len(points),
        )
        return points

    except Exception as exc:
        logger.exception("Qdrant query_points() failed for %s (limit=%r): %s", collection_name, limit, exc)
        return []


def _build_search_params() -> SearchParams:
    """
    Xây dựng SearchParams dùng chung cho mọi collection.

    [UPGRADE] hnsw_ef=256 (tăng từ 128): không còn quantization nên có
    thêm "budget" tốc độ để tăng ef, đổi lấy recall cao hơn.

    indexed_only=True — bỏ qua segment chưa kịp index (chỉ nên bật khi
    dataset không còn insert liên tục — xem cảnh báo trong query_collection()).

    KHÔNG còn QuantizationSearchParams — vì collection_fp32_hq không
    quantize, không có gì để "rescore". Nếu dùng lại collection SQ8
    (image_siglip_sq8) phải thêm lại tham số này, nếu không recall sẽ
    giảm âm thầm mà không có log/exception nào cảnh báo.
    """
    return SearchParams(
        hnsw_ef=256,
        indexed_only=True,
    )