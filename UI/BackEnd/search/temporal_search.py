"""
Stage-based Temporal Search Engine (Multi-Event Sequential KIS) for Video Keyframe Retrieval.

Sequential multi-event search matching a timeline of stages (Stage 0, Stage 1, ..., Stage N-1)
under strict chronological ordering: timestamp(E0) < timestamp(E1) < ... < timestamp(EN-1)
with support for:
- Multiple sub-queries per stage (Text, OCR, ASR, Image / Keyframe similarity) with individual weights
- Per-stage weighted Reciprocal Rank Fusion (RRF)
- Distance / Gap constraints between consecutive stages (max_gap_seconds)
- Trake-style Beam Search DP sequence alignment with Temporal Distance Penalty
- Trake-style Action-Distance NMS for extracting distinct video sequences
- Standardized output format (timeline, time_span, duration_sec, global_score, matched_sources)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

# Search engine hooks & keyframe manager
from search.keyframe_manager import (
    _TIMESTAMP_LOOKUP,
    _load_video_metadata,
    parse_video_frame_id,
    get_nearest_keyframe,
    _normalize_rel_image_path,
    KEYFRAME_ROOT,
)
from search.ocr_search import search_ocr
from search.asr_search import search_asr
from search.qdrant_search import query_collection, COLLECTION_SIGLIP
from models.siglip_encoder import encode_text_siglip, encode_images_siglip

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================
DEFAULT_RRF_K: int = 60
DEFAULT_MAX_WORKERS: int = 32
DEFAULT_CONTEXT_BONUS_WEIGHT: float = 0.15
DEFAULT_TOP_K_PER_STAGE: int = 800
DEFAULT_TOP_K_VIDEOS: int = 100
DEFAULT_FETCH_MULTIPLIER: int = 8
MIN_DISTINCT_EVENT_DIFF_SEC: float = 0.0

# Fusion cố định giữa Text và Image (0.75 text / 0.25 image)
TEXT_IMAGE_FUSION_WEIGHT_TEXT: float = 0.75
TEXT_IMAGE_FUSION_WEIGHT_IMAGE: float = 0.25

# Thread-safe SigLIP load lock
_siglip_load_lock = threading.Lock()


def _format_time_sec(ts_sec: Optional[float]) -> str:
    if ts_sec is None:
        return "--:--"
    total_sec = max(0, int(ts_sec))
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"


# ==============================================================================
# PYDANTIC DATA MODELS
# ==============================================================================
class StageSubQuery(BaseModel):
    """Sub-query item inside a single stage (supports multiple text, ocr, asr, image queries)."""
    id: Optional[str] = None
    type: str = Field("text", description="Loại tìm kiếm: 'text' | 'ocr' | 'asr' | 'object' | 'image' | 'keyframe'")
    text: Optional[str] = Field(None, description="Query text cho text/ocr/asr hoặc helper text cho image")
    query: Optional[str] = Field(None, description="Alias cho text hoặc video_id_frame_id")
    helper_text: Optional[str] = Field(None, description="Text bổ trợ đi kèm ảnh khi tìm theo ảnh (fusion 0.75/0.25)")
    video_id: Optional[str] = Field(None, description="Video ID cho Keyframe similarity")
    frame_id: Optional[str] = Field(None, description="Frame ID cho Keyframe similarity")
    weight: float = Field(1.0, description="Trọng số query item này trong RRF của stage")


class StageQuery(BaseModel):
    """Query model for a single stage in the chronological timeline."""
    stage_index: int = Field(..., description="Thứ tự stage, bắt đầu từ 0")
    queries: Optional[List[StageSubQuery]] = Field(None, description="Danh sách các sub-query linh hoạt trong stage này")
    max_gap_seconds: Optional[float] = Field(None, description="Khoảng cách tối đa tới stage tiếp theo (giây)")
    
    # Trọng số cấp Stage so với các stage khác (mặc định 1.0)
    stage_weight: float = Field(1.0, description="Trọng số của cả stage này so với các stage khác (mặc định 1.0)")

    # Legacy flat fields for backward compatibility
    text_query: Optional[str] = Field(None, description="Query semantic (text-to-image)")
    image_query: Optional[str] = Field(None, description="Query image similarity (video_id_frame_id)")
    helper_text: Optional[str] = Field(None, description="Text bổ trợ khi dùng image_query (fusion 0.75/0.25)")
    ocr_query: Optional[str] = Field(None, description="Query OCR (có thể để trống)")
    asr_query: Optional[str] = Field(None, description="Query ASR (có thể để trống)")
    weight_text: float = Field(1.0, description="Trọng số query text")
    weight_image: float = Field(1.0, description="Trọng số query image")
    weight_ocr: float = Field(1.0, description="Trọng số query OCR")
    weight_asr: float = Field(1.0, description="Trọng số query ASR")

    @model_validator(mode="after")
    def validate_has_query(self) -> "StageQuery":
        if self.queries:
            valid_sub = [
                q for q in self.queries
                if (q.type in ("image", "keyframe") and ((q.video_id and q.frame_id) or (q.text and q.text.strip()) or (q.query and q.query.strip())))
                or (q.type not in ("image", "keyframe") and bool((q.text and q.text.strip()) or (q.query and q.query.strip())))
            ]
            if not valid_sub:
                raise ValueError(f"Stage {self.stage_index} must have at least one non-empty sub-query in queries list.")
            return self

        has_text = bool(self.text_query and self.text_query.strip())
        has_image = bool(self.image_query and self.image_query.strip())
        has_ocr = bool(self.ocr_query and self.ocr_query.strip())
        has_asr = bool(self.asr_query and self.asr_query.strip())
        if not (has_text or has_image or has_ocr or has_asr):
            raise ValueError(
                f"Stage {self.stage_index} must specify at least one non-empty query (queries list, text_query, image_query, ocr_query, or asr_query)."
            )
        return self


class TemporalSearchRequest(BaseModel):
    """Stage-based temporal search request payload."""
    stages: List[StageQuery] = Field(..., description="Danh sách các stage theo thứ tự thời gian tăng dần")
    global_context: Optional[str] = Field(None, description="Mô tả bối cảnh toàn cục của video")
    context_bonus_weight: float = Field(DEFAULT_CONTEXT_BONUS_WEIGHT, description="Trọng số cộng điểm bối cảnh toàn cục (mặc định 0.15)")
    max_gap_seconds: Optional[float] = Field(None, description="Khoảng cách tối đa toàn cục giữa các stage liên tiếp")
    video_list: Optional[List[str]] = Field(None, description="Danh sách video_id cần lọc trước")
    top_k_videos: int = Field(DEFAULT_TOP_K_VIDEOS, description="Số lượng chuỗi video tối đa trả về")
    top_k_per_stage: int = Field(DEFAULT_TOP_K_PER_STAGE, description="Số lượng candidate tối đa giữ lại cho mỗi stage trước khi DP alignment")


class StageEventResult(BaseModel):
    """Result for a matched keyframe in a specific stage."""
    event_id: str = Field(..., description="Tên sự kiện / Stage (VD: 'Stage 1')")
    stage_index: int = Field(..., description="Chỉ số stage (0, 1, 2, ...)")
    video_id: str = Field(..., description="Video ID (VD: 'L21_V029')")
    keyframe_id: str = Field(..., description="Keyframe ID (VD: 'L21_V029_kf_0713')")
    frame_idx: int = Field(..., description="Frame index")
    timestamp: float = Field(..., description="Timestamp tính bằng giây")
    pts_time: Optional[float] = Field(None, description="PTS time (giây)")
    time_str: Optional[str] = Field(None, description="Định dạng thời gian MM:SS")
    image_path: str = Field(..., description="Đường dẫn ảnh tương đối")
    image_url: Optional[str] = Field(None, description="Đường dẫn URL ảnh")
    score: float = Field(..., description="Điểm relevance của keyframe tại stage này")
    matched_sources: List[str] = Field(default_factory=list, description="Danh sách nguồn khớp (text, ocr, asr, image)")


class VideoResult(BaseModel):
    """Result for an entire aligned temporal sequence of stages in a video."""
    video_id: str = Field(..., description="Video ID")
    sequence_id: int = Field(1, description="Thứ tự chuỗi khác biệt trong video (1, 2, 3...)")
    time_span: str = Field(..., description="Khoảng thời gian (VD: '00:15 -> 02:40')")
    duration_sec: float = Field(0.0, description="Độ dài chuỗi tính bằng giây")
    global_score: float = Field(..., description="Điểm tổng hợp chuỗi (tính cả context bonus)")
    context_bonus: float = Field(0.0, description="Điểm bonus từ bối cảnh toàn cục")
    timeline: List[StageEventResult] = Field(default_factory=list, description="Chuỗi các sự kiện khớp theo đúng thứ tự thời gian")
    frame_chain: List[StageEventResult] = Field(default_factory=list, description="Alias cho timeline")


class TemporalSearchResponse(BaseModel):
    """Response payload for temporal search."""
    status: str = Field("success", description="Trạng thái thực thi ('success' | 'error')")
    message: Optional[str] = Field(None, description="Thông điệp bổ sung")
    videos: List[VideoResult] = Field(default_factory=list, description="Danh sách video kết quả sắp xếp theo global_score giảm dần")


# ==============================================================================
# SUB-QUERY EXTRACTION & ENGINE SEARCH HELPERS
# ==============================================================================
def _extract_stage_sub_queries(stage: StageQuery) -> List[Dict[str, Any]]:
    """Trích xuất danh sách sub-query sạch từ StageQuery."""
    sub_queries: List[Dict[str, Any]] = []

    # 1. New dynamic queries list
    if stage.queries:
        for idx, sq in enumerate(stage.queries):
            q_type = (sq.type or "text").lower().strip()
            q_text = (sq.text or sq.query or "").strip()
            q_vid = (sq.video_id or "").strip()
            q_fid = (sq.frame_id or "").strip()
            q_helper = (sq.helper_text or "").strip()
            q_weight = float(sq.weight) if sq.weight is not None else 1.0

            if q_type in ("image", "keyframe"):
                raw_ref = q_text or (f"{q_vid}_{q_fid}" if q_vid and q_fid else (q_vid or q_fid))
                if raw_ref:
                    img_label = f"image_{idx}"
                    if q_helper:
                        txt_label = f"text_helper_{idx}"
                        sub_queries.append({
                            "type": "image",
                            "raw_query": raw_ref,
                            "video_id": q_vid,
                            "frame_id": q_fid,
                            "weight": q_weight,
                            "label": img_label,
                            "pair_text_key": txt_label
                        })
                        sub_queries.append({
                            "type": "text",
                            "text": q_helper,
                            "weight": q_weight,
                            "label": txt_label,
                            "pair_image_key": img_label
                        })
                    else:
                        sub_queries.append({
                            "type": "image",
                            "raw_query": raw_ref,
                            "video_id": q_vid,
                            "frame_id": q_fid,
                            "weight": q_weight,
                            "label": img_label
                        })
            elif q_type in ("ocr", "ocr_text"):
                if q_text:
                    sub_queries.append({
                        "type": "ocr",
                        "text": q_text,
                        "weight": q_weight,
                        "label": f"ocr_{idx}"
                    })
            elif q_type in ("asr", "audio"):
                if q_text:
                    sub_queries.append({
                        "type": "asr",
                        "text": q_text,
                        "weight": q_weight,
                        "label": f"asr_{idx}"
                    })
            else:
                if q_text:
                    sub_queries.append({
                        "type": "text",
                        "text": q_text,
                        "weight": q_weight,
                        "label": f"text_{idx}"
                    })

    # 2. Legacy fallback
    if not sub_queries:
        has_text = bool(stage.text_query and stage.text_query.strip())
        has_image = bool(stage.image_query and stage.image_query.strip())
        has_helper = bool(stage.helper_text and stage.helper_text.strip())
        effective_text = stage.text_query.strip() if has_text else (stage.helper_text.strip() if has_helper else "")

        if effective_text and has_image:
            sub_queries.append({
                "type": "text",
                "text": effective_text,
                "weight": float(stage.weight_text),
                "label": "text",
                "pair_image_key": "image"
            })
            sub_queries.append({
                "type": "image",
                "raw_query": stage.image_query.strip(),
                "weight": float(stage.weight_image),
                "label": "image",
                "pair_text_key": "text"
            })
        else:
            if effective_text:
                sub_queries.append({
                    "type": "text",
                    "text": effective_text,
                    "weight": float(stage.weight_text),
                    "label": "text"
                })
            if has_image:
                sub_queries.append({
                    "type": "image",
                    "raw_query": stage.image_query.strip(),
                    "weight": float(stage.weight_image),
                    "label": "image"
                })

        if stage.ocr_query and stage.ocr_query.strip():
            sub_queries.append({
                "type": "ocr",
                "text": stage.ocr_query.strip(),
                "weight": float(stage.weight_ocr),
                "label": "ocr"
            })
        if stage.asr_query and stage.asr_query.strip():
            sub_queries.append({
                "type": "asr",
                "text": stage.asr_query.strip(),
                "weight": float(stage.weight_asr),
                "label": "asr"
            })

    return sub_queries


def _normalize_rel_image_path(raw_path: str, video_id: str, keyframe_id: str) -> str:
    """Chuẩn hóa đường dẫn tương đối của ảnh."""
    if not raw_path:
        return f"{video_id}/{keyframe_id}.jpg"
    p = raw_path.replace("\\", "/").strip()
    if p.startswith("/"):
        p = p.lstrip("/")
    if p.startswith("images/"):
        p = p[7:]
    elif p.startswith("keyframes/"):
        p = p[10:]
    return p


# ==============================================================================
# SEARCH FUNCTIONS & TIMESTAMPS RESOLUTION
# ==============================================================================
def _ensure_siglip_loaded():
    """Thread-safe SigLIP load: chỉ load 1 lần dù nhiều thread gọi đồng thời."""
    from models.siglip_encoder import _siglip_model, load_siglip
    if _siglip_model is None:
        with _siglip_load_lock:
            from models.siglip_encoder import _siglip_model as _m2
            if _m2 is None:
                load_siglip()


def search_semantic(query: str, top_k: int = 100, video_list: Optional[List[str]] = None) -> List[dict]:
    """Search keyframes using SigLIP text embedding against Qdrant vector database."""
    if not query or not query.strip():
        return []
    try:
        t0 = time.perf_counter()
        _ensure_siglip_loaded()
        emb = encode_text_siglip(query.strip())
        pts = query_collection(embedding=emb, collection_name=COLLECTION_SIGLIP, limit=top_k, video_list=video_list)
        logger.debug("[SemSearch] '%s': %.3fs returned=%d", query[:40], time.perf_counter() - t0, len(pts))
        results = []
        for pt in pts:
            if pt.payload and pt.payload.get("video_id"):
                results.append({
                    "video_id": str(pt.payload.get("video_id")),
                    "keyframe_id": str(pt.payload.get("keyframe_id", "")),
                    "frame_idx": int(pt.payload.get("frame_idx", 0)),
                    "timestamp": pt.payload.get("timestamp"),
                    "image_path": str(pt.payload.get("image_path", "")),
                    "score": float(pt.score) if pt.score is not None else 0.0
                })
        return results
    except Exception as e:
        logger.error(f"[Semantic Search] Error querying '{query[:40]}': {e}")
        return []


def search_ocr_safe(query: str, top_k: int = 100, video_list: Optional[List[str]] = None) -> List[dict]:
    """Search keyframes using Elasticsearch OCR text."""
    if not query or not query.strip():
        return []
    try:
        t0 = time.perf_counter()
        res = search_ocr(query.strip(), top_k, video_list)
        logger.debug("[OCRSearch] '%s': %.3fs returned=%d", query[:40], time.perf_counter() - t0, len(res or []))
        return res or []
    except Exception as e:
        logger.error(f"[OCR Search] Error querying '{query[:40]}': {e}")
        return []


def search_asr_safe(query: str, top_k: int = 100, video_list: Optional[List[str]] = None) -> List[dict]:
    """Search keyframes using Elasticsearch ASR transcript."""
    if not query or not query.strip():
        return []
    try:
        t0 = time.perf_counter()
        res = search_asr(query.strip(), top_k, video_list)
        logger.debug("[ASRSearch] '%s': %.3fs returned=%d", query[:40], time.perf_counter() - t0, len(res or []))
        return res or []
    except Exception as e:
        logger.error(f"[ASR Search] Error querying '{query[:40]}': {e}")
        return []


def search_image_similarity_safe(
    video_id: str,
    frame_id: Optional[str] = None,
    top_k: int = 100,
    video_list: Optional[List[str]] = None
) -> List[dict]:
    """Search keyframes using SigLIP image similarity against Qdrant vector database."""
    clean_combined = f"{video_id}_{frame_id}" if (video_id and frame_id) else (video_id or frame_id or "")
    parsed_vid, parsed_fidx, parsed_kfid = parse_video_frame_id(clean_combined)
    if not parsed_vid and video_id:
        parsed_vid = video_id.strip()
    if parsed_fidx is None and frame_id and frame_id.strip().isdigit():
        parsed_fidx = int(frame_id.strip())

    clean_vid = parsed_vid.replace(".mp4", "").strip()
    if not clean_vid:
        return []
    try:
        from PIL import Image

        nearest_kf = get_nearest_keyframe(clean_vid, parsed_fidx, parsed_kfid)
        img_path = None
        if nearest_kf and nearest_kf.get("image_path"):
            cand = os.path.join(KEYFRAME_ROOT, nearest_kf["image_path"])
            if os.path.isfile(cand):
                img_path = cand

        if not img_path:
            kfid_cand = parsed_kfid or (f"{clean_vid}_kf_{parsed_fidx:04d}" if parsed_fidx is not None else f"{clean_vid}_kf_0001")
            cand1 = os.path.join(KEYFRAME_ROOT, clean_vid, f"{kfid_cand}.jpg")
            if os.path.isfile(cand1):
                img_path = cand1

        if not img_path or not os.path.isfile(img_path):
            logger.warning("Stage image search: cannot find image for video=%s frame=%r", clean_vid, parsed_fidx)
            return []

        t0 = time.perf_counter()
        _ensure_siglip_loaded()
        with Image.open(img_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            img_emb = encode_images_siglip([rgb_img])[0]

        pts = query_collection(embedding=img_emb, collection_name=COLLECTION_SIGLIP, limit=top_k, video_list=video_list)
        logger.debug("[ImageSearch] video=%s frame=%r: %.3fs returned=%d", clean_vid, parsed_fidx, time.perf_counter() - t0, len(pts))

        results = []
        for pt in pts:
            if pt.payload and pt.payload.get("video_id"):
                results.append({
                    "video_id": str(pt.payload.get("video_id")),
                    "keyframe_id": str(pt.payload.get("keyframe_id", "")),
                    "frame_idx": int(pt.payload.get("frame_idx", 0)),
                    "timestamp": pt.payload.get("timestamp"),
                    "image_path": str(pt.payload.get("image_path", "")),
                    "score": float(pt.score) if pt.score is not None else 0.0
                })
        return results
    except Exception as e:
        logger.error(f"[Image Similarity Search] Error for video={video_id}: {e}")
        return []


def _batch_resolve_timestamps(
    doc_keys: List[Tuple[str, int, Optional[str]]]
) -> Dict[Tuple[str, int], float]:
    """BATCH timestamp lookup — O(1) dict lookup mỗi frame thay vì N+1 I/O."""
    result: Dict[Tuple[str, int], float] = {}
    missing_by_video: Dict[str, List[Tuple[int, Optional[str]]]] = {}

    for video_id, frame_idx, keyframe_id in doc_keys:
        ts_ms = None
        if keyframe_id and keyframe_id in _TIMESTAMP_LOOKUP:
            ts_ms = _TIMESTAMP_LOOKUP[keyframe_id]
        else:
            composite_key = f"{video_id}:{frame_idx}"
            if composite_key in _TIMESTAMP_LOOKUP:
                ts_ms = _TIMESTAMP_LOOKUP[composite_key]

        if ts_ms is not None:
            result[(video_id, frame_idx)] = round(float(ts_ms) / 1000.0, 3)
        else:
            if video_id not in missing_by_video:
                missing_by_video[video_id] = []
            missing_by_video[video_id].append((frame_idx, keyframe_id))

    if missing_by_video:
        for vid, frames_info in missing_by_video.items():
            kf_data = get_keyframes_data(vid)
            idx_map = {}
            for item in kf_data:
                f_idx = item.get("frame_idx")
                pts = item.get("pts_time")
                if f_idx is not None and pts is not None:
                    idx_map[int(f_idx)] = float(pts)

            for f_idx, k_id in frames_info:
                if f_idx in idx_map:
                    result[(vid, f_idx)] = round(idx_map[f_idx], 3)
                else:
                    ts = get_keyframe_timestamp(vid, k_id, f_idx)
                    result[(vid, f_idx)] = round(float(ts) / 1000.0, 3)

    return result


# ==============================================================================
# PIPELINE STEPS
# ==============================================================================
def _search_all_stages(
    request: TemporalSearchRequest,
    fetch_limit: int,
    executor: Optional[ThreadPoolExecutor] = None
) -> Tuple[Dict[int, Dict[str, dict]], List[dict]]:
    """
    BƯỚC 1 - Parallel Multi-Search:
    Tìm kiếm song song tất cả các sub-queries của tất cả các stages & global context.
    """
    should_close_exec = False
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS)
        should_close_exec = True

    try:
        futures = {}

        # 1. Submit all stage subqueries
        for stage in request.stages:
            s_idx = stage.stage_index
            sub_queries = _extract_stage_sub_queries(stage)

            for sq in sub_queries:
                q_key = sq["label"]
                q_label = sq["type"]
                q_weight = sq["weight"]
                q_extra = {
                    k: sq[k]
                    for k in ("pair_text_key", "pair_image_key")
                    if k in sq
                }

                if sq["type"] == "text":
                    fut = executor.submit(search_semantic, sq["text"], fetch_limit, request.video_list)
                    futures[fut] = ("stage", s_idx, q_key, q_label, q_weight, q_extra)
                elif sq["type"] == "ocr":
                    fut = executor.submit(search_ocr_safe, sq["text"], fetch_limit, request.video_list)
                    futures[fut] = ("stage", s_idx, q_key, q_label, q_weight, q_extra)
                elif sq["type"] == "asr":
                    fut = executor.submit(search_asr_safe, sq["text"], fetch_limit, request.video_list)
                    futures[fut] = ("stage", s_idx, q_key, q_label, q_weight, q_extra)
                elif sq["type"] == "image":
                    fut = executor.submit(search_image_similarity_safe, sq["raw_query"], None, fetch_limit, request.video_list)
                    futures[fut] = ("stage", s_idx, q_key, q_label, q_weight, q_extra)

        # 2. Submit global context query
        has_context = bool(request.global_context and request.global_context.strip())
        if has_context:
            fut = executor.submit(search_semantic, request.global_context.strip(), fetch_limit, request.video_list)
            futures[fut] = ("global", -1, "semantic", "global_context", 1.0, {})

        # 3. Gather results
        stage_engine_results: Dict[int, Dict[str, dict]] = {s.stage_index: {} for s in request.stages}
        global_context_results: List[dict] = []

        for fut in as_completed(futures):
            q_type, s_idx, run_key, run_label, weight, extra_meta = futures[fut]
            try:
                raw_items = fut.result() or []
                if q_type == "stage":
                    stage_engine_results[s_idx][run_key] = {
                        "results": raw_items,
                        "label": run_label,
                        "weight": weight,
                        **extra_meta
                    }
                elif q_type == "global":
                    global_context_results = raw_items
            except Exception as e:
                logger.error(f"[Multi-Search] Error gathering future ({q_type}, stage={s_idx}, run={run_key}): {e}")

        return stage_engine_results, global_context_results
    finally:
        if should_close_exec:
            executor.shutdown(wait=False)


def _fuse_text_image(text_results: List[dict], image_results: List[dict]) -> List[dict]:
    """
    Fusion text_query và image_query trong CÙNG 1 stage bằng công thức tuyến tính 
    cố định: f_fusion = 0.75*f_text + 0.25*f_image.
    Trả về list đã fusion, dùng làm 1 "run" duy nhất để đưa tiếp vào RRF chung 
    của stage với OCR/ASR.
    """
    if not text_results and not image_results:
        return []
    if not text_results:
        return image_results
    if not image_results:
        return text_results

    doc_map: Dict[Tuple[str, int], dict] = {}

    for item in text_results:
        vid = str(item.get("video_id", ""))
        if not vid:
            continue
        fidx = int(item.get("frame_idx", 0))
        key = (vid, fidx)
        score = float(item.get("score", 0.0))
        doc_map[key] = {
            "video_id": vid,
            "frame_idx": fidx,
            "keyframe_id": item.get("keyframe_id"),
            "image_path": item.get("image_path", ""),
            "timestamp": item.get("timestamp"),
            "text_score": score,
            "image_score": 0.0,
        }

    for item in image_results:
        vid = str(item.get("video_id", ""))
        if not vid:
            continue
        fidx = int(item.get("frame_idx", 0))
        key = (vid, fidx)
        score = float(item.get("score", 0.0))
        if key in doc_map:
            doc_map[key]["image_score"] = score
            if not doc_map[key].get("image_path") and item.get("image_path"):
                doc_map[key]["image_path"] = item.get("image_path")
            if not doc_map[key].get("keyframe_id") and item.get("keyframe_id"):
                doc_map[key]["keyframe_id"] = item.get("keyframe_id")
            if doc_map[key].get("timestamp") is None and item.get("timestamp") is not None:
                doc_map[key]["timestamp"] = item.get("timestamp")
        else:
            doc_map[key] = {
                "video_id": vid,
                "frame_idx": fidx,
                "keyframe_id": item.get("keyframe_id"),
                "image_path": item.get("image_path", ""),
                "timestamp": item.get("timestamp"),
                "text_score": 0.0,
                "image_score": score,
            }

    fused_results = []
    for doc in doc_map.values():
        fused_score = (
            TEXT_IMAGE_FUSION_WEIGHT_TEXT * doc["text_score"]
            + TEXT_IMAGE_FUSION_WEIGHT_IMAGE * doc["image_score"]
        )
        fused_results.append({
            "video_id": doc["video_id"],
            "frame_idx": doc["frame_idx"],
            "keyframe_id": doc.get("keyframe_id"),
            "image_path": doc.get("image_path", ""),
            "timestamp": doc.get("timestamp"),
            "score": round(fused_score, 6),
        })

    fused_results.sort(key=lambda x: x["score"], reverse=True)
    return fused_results


def _fuse_engine_results(
    stages: List[StageQuery],
    stage_engine_results: Dict[int, Dict[str, dict]],
    top_k_per_stage: int = DEFAULT_TOP_K_PER_STAGE,
    rrf_k: int = DEFAULT_RRF_K
) -> Dict[int, List[StageEventResult]]:
    """
    BƯỚC 2 - RRF Fusion cho từng stage:
    1. Khi stage có cả text và image -> Fusion riêng f_fusion = 0.75*text + 0.25*image làm 1 run duy nhất.
    2. Dung hợp RRF chuẩn hóa giữa các run trong stage (VD: RRF(fusion(text,image), ocr, asr)).
    Điểm mỗi run được chuẩn hóa theo (rrf_k + 1.0) / (rrf_k + rank + 1) để nằm trong khoảng [0, weight].
    Ghi nhận matched_sources cho mỗi candidate.
    """
    stage_top_docs: Dict[int, List[Tuple[Tuple[str, int], float, dict, List[str]]]] = {}

    for stage in stages:
        s_idx = stage.stage_index
        engine_dict = stage_engine_results.get(s_idx, {})

        if not engine_dict:
            stage_top_docs[s_idx] = []
            continue

        # 1. Text + Image Fusion (0.75 / 0.25)
        processed_engine_dict: Dict[str, dict] = {}
        paired_keys = set()

        # Tìm các cặp có liên kết trực tiếp (pair_text_key / pair_image_key)
        for run_key, run_data in engine_dict.items():
            if run_key in paired_keys:
                continue
            pair_img = run_data.get("pair_image_key")
            pair_txt = run_data.get("pair_text_key")
            if pair_img and pair_img in engine_dict and pair_img not in paired_keys:
                txt_items = run_data.get("results", [])
                img_items = engine_dict[pair_img].get("results", [])
                fused_items = _fuse_text_image(txt_items, img_items)
                fused_key = f"fusion_{run_key}_{pair_img}"
                processed_engine_dict[fused_key] = {
                    "results": fused_items,
                    "label": "text_image_fusion",
                    "weight": run_data.get("weight", 1.0)
                }
                paired_keys.add(run_key)
                paired_keys.add(pair_img)
            elif pair_txt and pair_txt in engine_dict and pair_txt not in paired_keys:
                img_items = run_data.get("results", [])
                txt_items = engine_dict[pair_txt].get("results", [])
                fused_items = _fuse_text_image(txt_items, img_items)
                fused_key = f"fusion_{pair_txt}_{run_key}"
                processed_engine_dict[fused_key] = {
                    "results": fused_items,
                    "label": "text_image_fusion",
                    "weight": run_data.get("weight", 1.0)
                }
                paired_keys.add(run_key)
                paired_keys.add(pair_txt)

        # Nếu có 1 text và 1 image độc lập trong stage -> tự động fuse
        unpaired_text_keys = [k for k, v in engine_dict.items() if k not in paired_keys and v.get("label") == "text"]
        unpaired_image_keys = [k for k, v in engine_dict.items() if k not in paired_keys and v.get("label") == "image"]

        if len(unpaired_text_keys) == 1 and len(unpaired_image_keys) == 1:
            t_k = unpaired_text_keys[0]
            i_k = unpaired_image_keys[0]
            txt_items = engine_dict[t_k].get("results", [])
            img_items = engine_dict[i_k].get("results", [])
            fused_items = _fuse_text_image(txt_items, img_items)
            fused_key = f"fusion_{t_k}_{i_k}"
            processed_engine_dict[fused_key] = {
                "results": fused_items,
                "label": "text_image_fusion",
                "weight": engine_dict[t_k].get("weight", 1.0)
            }
            paired_keys.add(t_k)
            paired_keys.add(i_k)

        # Các run còn lại (hoặc khi chỉ có 1 trong 2: text riêng hoặc image riêng, hoặc ocr, asr)
        for run_key, run_data in engine_dict.items():
            if run_key not in paired_keys:
                processed_engine_dict[run_key] = run_data

        # 2. RRF dung hợp tất cả các run đã xử lý
        fused_scores: Dict[Tuple[str, int], float] = {}
        frame_metadata: Dict[Tuple[str, int], dict] = {}
        frame_sources: Dict[Tuple[str, int], Set[str]] = {}

        for run_key, run_data in processed_engine_dict.items():
            items = run_data.get("results", [])
            weight = float(run_data.get("weight", 1.0))
            source_label = run_data.get("label", run_key)

            for rank_idx, item in enumerate(items):
                vid = str(item.get("video_id", ""))
                if not vid:
                    continue
                fidx = int(item.get("frame_idx", 0))
                doc_key = (vid, fidx)

                # Normalized RRF score: Top-1 has score = weight * 1.0
                rrf_score = weight * ((rrf_k + 1.0) / (rrf_k + (rank_idx + 1)))
                fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + rrf_score

                if doc_key not in frame_sources:
                    frame_sources[doc_key] = set()
                frame_sources[doc_key].add(source_label)

                if doc_key not in frame_metadata:
                    frame_metadata[doc_key] = {
                        "video_id": vid,
                        "frame_idx": fidx,
                        "keyframe_id": item.get("keyframe_id"),
                        "image_path": item.get("image_path", ""),
                        "timestamp": item.get("timestamp")
                    }

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_per_stage]
        stage_top_docs[s_idx] = [
            (doc_key, score, frame_metadata[doc_key], sorted(list(frame_sources.get(doc_key, []))))
            for doc_key, score in sorted_docs
        ]

    # Batch timestamp resolution
    batch_lookup_keys: List[Tuple[str, int, Optional[str]]] = []
    for s_idx, docs in stage_top_docs.items():
        for doc_key, score, meta, sources in docs:
            if meta.get("timestamp") is None:
                vid = meta["video_id"]
                fidx = meta["frame_idx"]
                kfid = meta.get("keyframe_id")
                batch_lookup_keys.append((vid, fidx, kfid))

    ts_batch: Dict[Tuple[str, int], float] = {}
    if batch_lookup_keys:
        t_ts = time.perf_counter()
        ts_batch = _batch_resolve_timestamps(batch_lookup_keys)
        logger.debug("[Step2] Batch timestamp lookup: %d frames in %.3fs", len(batch_lookup_keys), time.perf_counter() - t_ts)

    # Build StageEventResult items
    stage_candidates: Dict[int, List[StageEventResult]] = {}
    for stage in stages:
        s_idx = stage.stage_index
        docs = stage_top_docs.get(s_idx, [])
        candidates: List[StageEventResult] = []

        for doc_key, score, meta, sources in docs:
            vid = meta["video_id"]
            fidx = meta["frame_idx"]
            kfid = meta.get("keyframe_id") or f"{vid}_kf_{fidx:04d}"
            raw_img = meta.get("image_path") or ""
            img_path = _normalize_rel_image_path(raw_img, vid, kfid)

            raw_ts = meta.get("timestamp")
            if raw_ts is not None:
                ts = float(raw_ts) if raw_ts <= 10000 else round(float(raw_ts) / 1000.0, 3)
            else:
                ts = ts_batch.get((vid, fidx), 0.0)

            candidates.append(StageEventResult(
                event_id=f"Stage {s_idx + 1}",
                stage_index=s_idx,
                video_id=vid,
                keyframe_id=kfid,
                frame_idx=fidx,
                timestamp=ts,
                pts_time=ts,
                time_str=_format_time_sec(ts),
                image_path=img_path,
                image_url=f"/images/{img_path}" if img_path else None,
                score=round(score, 6),
                matched_sources=sources
            ))

        stage_candidates[s_idx] = candidates

    return stage_candidates


def _group_and_prune_by_video(
    stages: List[StageQuery],
    stage_candidates: Dict[int, List[StageEventResult]]
) -> Dict[str, Dict[int, List[StageEventResult]]]:
    """
    BƯỚC 3 - Group theo Video & Prune (TRAKE requirement):
    Chỉ giữ lại những video có chứa candidate ở TẤT CẢ các stage.
    """
    stage_indices = [s.stage_index for s in stages]
    if not stage_indices:
        return {}

    all_videos: Set[str] = set()
    for s_idx in stage_indices:
        for cand in stage_candidates.get(s_idx, []):
            all_videos.add(cand.video_id)

    candidates_by_video: Dict[str, Dict[int, List[StageEventResult]]] = {
        vid: {s_idx: [] for s_idx in stage_indices} for vid in all_videos
    }

    for s_idx in stage_indices:
        for cand in stage_candidates.get(s_idx, []):
            candidates_by_video[cand.video_id][s_idx].append(cand)

    valid_videos: Dict[str, Dict[int, List[StageEventResult]]] = {
        vid: cands for vid, cands in candidates_by_video.items()
        if all(len(cands[s_idx]) > 0 for s_idx in stage_indices)
    }

    logger.info(
        "[TRAKE/Stage Grouping] Tổng %d video tiềm năng, %d video hợp lệ có đủ %d stage.",
        len(all_videos), len(valid_videos), len(stage_indices)
    )
    return valid_videos


def _beam_search_dp_alignment(
    video_id: str,
    stages: List[StageQuery],
    video_candidates: Dict[int, List[StageEventResult]],
    max_gap_seconds: Optional[float] = None,
    context_bonus: float = 0.0,
    context_bonus_weight: float = DEFAULT_CONTEXT_BONUS_WEIGHT,
    max_sequences: int = 3,
    beam_width: int = 6,
    min_score_ratio: float = 0.55
) -> List[VideoResult]:
    """
    BƯỚC 4 & 5 - Beam Search DP + Action-Distance NMS (chuẩn TRAKE):
    - Sắp xếp candidate mỗi stage theo timestamp tăng dần.
    - Beam Search DP giữ lại Top-B đường đi tại mỗi trạng thái.
    - Kiểm tra thứ tự thời gian nghiêm ngặt t_{i-1} < t_i và ràng buộc max_gap_seconds.
    - Áp dụng Temporal Distance Penalty cho khoảng cách > 30s.
    - Action-Distance NMS chọn tối đa Top-K chuỗi thực sự khác biệt (cách nhau >= 10s trên ít nhất 1 stage).
    - Tính global_score = round(max(0.0, p_score / n_stages) + (context_bonus * context_bonus_weight), 4).
    """
    n_stages = len(stages)
    if n_stages == 0:
        return []

    sorted_cands: Dict[int, List[StageEventResult]] = {}
    for s in stages:
        s_idx = s.stage_index
        cands = list(video_candidates.get(s_idx, []))
        cands.sort(key=lambda x: x.timestamp)
        sorted_cands[s_idx] = cands

    s0_idx = stages[0].stage_index
    if not sorted_cands.get(s0_idx):
        return []

    # Trọng số Stage 0
    w0 = float(stages[0].stage_weight if getattr(stages[0], 'stage_weight', None) is not None else 1.0)

    # dp: danh sách các layer, mỗi layer gồm list các dict {'cand': StageEventResult, 'paths': List[path]}
    dp: List[List[Dict[str, Any]]] = []

    # Khởi tạo Stage 0 (nhân với stage_weight của Stage 0)
    init_layer = []
    for cand in sorted_cands[s0_idx]:
        init_layer.append({
            'cand': cand,
            'paths': [{
                'score': w0 * cand.score,
                'timeline': [cand]
            }]
        })
    dp.append(init_layer)

    # Duyệt Beam Search qua Stage 1 -> N-1
    for i in range(1, n_stages):
        curr_stage = stages[i]
        curr_stage_idx = curr_stage.stage_index
        curr_weight = float(curr_stage.stage_weight if getattr(curr_stage, 'stage_weight', None) is not None else 1.0)
        prev_stage = stages[i - 1]
        
        # Max gap giữa stage i-1 và stage i (ưu tiên cấu hình trên stage i-1, sau đó global max_gap_seconds)
        stage_max_gap = prev_stage.max_gap_seconds if prev_stage.max_gap_seconds is not None else max_gap_seconds

        curr_cands = sorted_cands.get(curr_stage_idx, [])
        if not curr_cands:
            return []

        prev_layer = dp[i - 1]
        curr_layer: List[Dict[str, Any]] = []

        for cand_curr in curr_cands:
            curr_ts = cand_curr.timestamp
            incoming_paths = []

            for prev_entry in prev_layer:
                prev_cand: StageEventResult = prev_entry['cand']
                prev_ts = prev_cand.timestamp

                if prev_ts >= curr_ts:
                    break  # vì prev_layer đã sort theo timestamp

                dt_sec = curr_ts - prev_ts

                # Kiểm tra ràng buộc khoảng cách tối đa giữa 2 stage (khi người dùng cấu hình trên UI)
                if stage_max_gap is not None and dt_sec > stage_max_gap:
                    continue

                for p in prev_entry['paths']:
                    if p['score'] <= -float('inf'):
                        continue
                    # Điểm tích lũy có nhân trọng số của stage hiện tại
                    new_score = p['score'] + (curr_weight * cand_curr.score)
                    incoming_paths.append({
                        'score': new_score,
                        'timeline': p['timeline'] + [cand_curr]
                    })

            incoming_paths.sort(key=lambda x: x['score'], reverse=True)
            top_paths = incoming_paths[:beam_width]

            if top_paths:
                curr_layer.append({
                    'cand': cand_curr,
                    'paths': top_paths
                })

        if not curr_layer:
            return []
        dp.append(curr_layer)

    # Thu thập toàn bộ các path hoàn chỉnh tại stage cuối cùng
    all_final_paths = []
    for entry in dp[-1]:
        all_final_paths.extend(entry['paths'])

    if not all_final_paths:
        return []

    all_final_paths.sort(key=lambda x: x['score'], reverse=True)

    # Áp dụng NMS (chọn tối đa max_sequences chuỗi không trùng lặp hoàn toàn frame)
    selected_sequences: List[VideoResult] = []
    best_score = all_final_paths[0]['score']
    total_stage_weight = sum(float(s.stage_weight if getattr(s, 'stage_weight', None) is not None else 1.0) for s in stages)

    for path_entry in all_final_paths:
        p_score = path_entry['score']
        if p_score < best_score * min_score_ratio or p_score <= 0:
            break

        tl: List[StageEventResult] = path_entry['timeline']
        tl_frames = [f.frame_idx for f in tl]

        # Kiểm tra tính khác biệt so với các chuỗi đã chọn (không trùng lặp toàn bộ frame)
        is_duplicate = False
        for sel in selected_sequences:
            sel_frames = [f.frame_idx for f in sel.timeline]
            if tl_frames == sel_frames:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        t_start = tl[0].timestamp
        t_end = tl[-1].timestamp
        duration_sec = round(max(0.0, t_end - t_start), 1)
        time_span = f"{_format_time_sec(t_start)} -> {_format_time_sec(t_end)}"

        # Weighted average score chia cho tổng stage_weight
        weighted_avg_score = p_score / float(total_stage_weight) if total_stage_weight > 0 else 0.0
        bonus_val = context_bonus * context_bonus_weight
        global_score = round(max(0.0, weighted_avg_score + bonus_val), 4)

        seq_id = len(selected_sequences) + 1
        selected_sequences.append(VideoResult(
            video_id=video_id,
            sequence_id=seq_id,
            time_span=time_span,
            duration_sec=duration_sec,
            global_score=global_score,
            context_bonus=round(context_bonus, 4),
            timeline=tl,
            frame_chain=tl
        ))

        if len(selected_sequences) >= max_sequences:
            break

    return selected_sequences


def solve_temporal_search(request: TemporalSearchRequest) -> TemporalSearchResponse:
    """
    BƯỚC 6 - Main Entry Point for Stage-based Temporal Sequential Search:
    1. Parallel Multi-Search across stages & global context.
    2. Per-stage RRF Fusion & batch timestamp resolution.
    3. Trake-style grouping & candidate validation.
    4. Trake-style Beam Search DP + Action-Distance NMS.
    5. Ranking and top_k_videos truncation.
    """
    total_start = time.perf_counter()
    if not request.stages:
        return TemporalSearchResponse(videos=[])

    fetch_limit = max(800, request.top_k_videos * DEFAULT_FETCH_MULTIPLIER)
    stage_top_k = max(800, request.top_k_per_stage)

    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        # BƯỚC 1: Multi-Search
        t1_start = time.perf_counter()
        stage_engine_results, global_context_results = _search_all_stages(
            request=request,
            fetch_limit=fetch_limit,
            executor=executor
        )
        t1_elapsed = time.perf_counter() - t1_start

        # BƯỚC 2: RRF Fusion
        t2_start = time.perf_counter()
        stage_candidates = _fuse_engine_results(
            stages=request.stages,
            stage_engine_results=stage_engine_results,
            top_k_per_stage=stage_top_k,
            rrf_k=DEFAULT_RRF_K
        )

        context_bonus_by_video: Dict[str, float] = {}
        if global_context_results:
            for rank_idx, item in enumerate(global_context_results):
                vid = str(item.get("video_id", ""))
                if vid:
                    rrf_context_score = (DEFAULT_RRF_K + 1.0) / (DEFAULT_RRF_K + (rank_idx + 1))
                    if rrf_context_score > context_bonus_by_video.get(vid, 0.0):
                        context_bonus_by_video[vid] = rrf_context_score
        t2_elapsed = time.perf_counter() - t2_start

        # BƯỚC 3: Group & Validate candidates
        t3_start = time.perf_counter()
        grouped_videos = _group_and_prune_by_video(
            stages=request.stages,
            stage_candidates=stage_candidates
        )
        t3_elapsed = time.perf_counter() - t3_start

        # BƯỚC 4 & 5: Beam Search DP Alignment & NMS
        t45_start = time.perf_counter()
        video_results: List[VideoResult] = []

        def _process_single_video(vid: str, v_cands: Dict[int, List[StageEventResult]]) -> List[VideoResult]:
            return _beam_search_dp_alignment(
                video_id=vid,
                stages=request.stages,
                video_candidates=v_cands,
                max_gap_seconds=request.max_gap_seconds,
                context_bonus=context_bonus_by_video.get(vid, 0.0),
                context_bonus_weight=request.context_bonus_weight,
                max_sequences=3,
                beam_width=6,
                min_score_ratio=0.55
            )

        if len(grouped_videos) > 10:
            fut_map = {
                executor.submit(_process_single_video, vid, cands): vid
                for vid, cands in grouped_videos.items()
            }
            for fut in as_completed(fut_map):
                try:
                    res_list = fut.result()
                    if res_list:
                        video_results.extend(res_list)
                except Exception as e:
                    logger.error(f"[DP Alignment] Error processing video {fut_map[fut]}: {e}")
        else:
            for vid, cands in grouped_videos.items():
                res_list = _process_single_video(vid, cands)
                if res_list:
                    video_results.extend(res_list)

        t45_elapsed = time.perf_counter() - t45_start

    # BƯỚC 6: Ranking & Truncation
    t6_start = time.perf_counter()
    video_results.sort(key=lambda x: x.global_score, reverse=True)
    final_videos = video_results[:request.top_k_videos]
    t6_elapsed = time.perf_counter() - t6_start
    total_elapsed = time.perf_counter() - total_start

    logger.info(
        "[STAGE/TRAKE TEMPORAL SEARCH] Done in %.4fs | Step1_Search=%.4fs | Step2_RRF=%.4fs | "
        "Step3_Prune=%.4fs (%d vids kept) | Step45_DP=%.4fs | Step6_Rank=%.4fs | Output=%d sequences",
        total_elapsed, t1_elapsed, t2_elapsed, t3_elapsed, len(grouped_videos),
        t45_elapsed, t6_elapsed, len(final_videos)
    )

    return TemporalSearchResponse(
        status="success",
        message="Đã trích xuất chuỗi thời gian thành công",
        videos=final_videos
    )


# ==============================================================================
# FASTAPI ROUTER
# ==============================================================================
temporal_router = APIRouter(prefix="/api/v1/temporal-search", tags=["Temporal Search"])


@temporal_router.post("/solve", response_model=TemporalSearchResponse)
def solve_temporal_search_endpoint(request: TemporalSearchRequest):
    """
    Solve stage-based temporal sequential search request across video keyframes.
    """
    try:
        return solve_temporal_search(request)
    except Exception as e:
        logger.exception("Error in /api/v1/temporal-search/solve:")
        raise HTTPException(status_code=500, detail=str(e))
