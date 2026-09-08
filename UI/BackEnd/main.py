"""
FastAPI backend: Text-to-Image Search via PE-Core-L + SigLIP2 + Relative Score Fusion.

Run with:
    conda activate aic2026_backend
    cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd
    uvicorn main:app --host 0.0.0.0 --port 8888

Env:
    CUDA_VISIBLE_DEVICES is set to "0" at startup (see below).
    Models are loaded once during lifespan, not per-request.

CHANGELOG (perf fix):
    - [FIX] Bước build response (gán timestamp cho từng keyframe) trước đây
      chạy ĐỒNG BỘ ngay trên event loop chính, gọi get_keyframe_timestamp()
      có thể phải đọc file CSV trên NAS mỗi khi cache-miss theo video_id.
      Với top_k lớn / kết quả trải nhiều video, việc này cộng dồn thành
      block đáng kể toàn bộ server (không chỉ chậm riêng request đó).
      -> Sửa: preload toàn bộ metadata keyframe vào RAM lúc startup
         (lifespan) để runtime chỉ còn tra cứu dict, không còn I/O.
      -> Đồng thời vẫn chạy qua run_in_executor + gather cho an toàn,
         phòng trường hợp preload thiếu video mới phát sinh sau khi
         server đã chạy (cache-miss hiếm, không còn chặn event loop).
    - [CLEANUP] Bỏ 2 hàm helper _encode_both / _query_both — không còn
      được gọi ở đâu trong file (dead code, dễ gây hiểu nhầm khi debug).
    - [FIX] time.perf_counter() mốc t0 đo thời gian Qdrant trước đây bị đặt
      ở module-level của qdrant_search.py (chỉ chạy 1 lần lúc import) nên
      log sai. Đã dời vào bên trong query_collection() ở file đó
      (không thuộc phạm vi file main.py này).

CHANGELOG (fusion revert):
    - [REVERT] Đã thử đổi từ reciprocal_rank_fusion (RRF) sang
      relative_score_fusion (min-max normalize) cho /api/search, nhưng
      revert lại RRF vì rủi ro: min-max quá nhạy outlier trên điểm OCR/ASR
      (Elasticsearch score không bound, đuôi dài dễ có outlier bùng nổ
      kéo giãn s_max, nén toàn bộ phần còn lại của list sát về 0), và range
      normalize phụ thuộc fetch depth (top_k khác nhau -> range khác nhau
      -> cùng 1 document ra điểm khác nhau). RRF (rank-based, adaptive k
      theo nguồn qua RRF_K_BY_SOURCE) không có 2 vấn đề này.
"""

# ── Fix visible GPU before any torch/transformers import ──────────────────────
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from models.bge_reranker import load_bge_reranker, rerank_candidates
from models.pe_core_encoder import encode_text_pecore, load_pecore
from models.siglip_encoder import encode_text_siglip, encode_images_siglip, load_siglip
from schemas import (
    HealthResponse,
    ImageResult,
    KeyframeMeta,
    SearchRequest,
    SearchResponse,
    VideoKeyframesResponse,
)
from search.keyframe_manager import (
    get_keyframe_timestamp,
    resolve_keyframe_metadata,
    get_video_keyframes,
    get_video_fps,
    get_video_info,
    get_all_videos_fps,
    _normalize_rel_image_path,
    find_video_path,
    parse_video_frame_id,
    get_nearest_keyframe,
)
from search.qdrant_search import (
    COLLECTION_PECORE,
    COLLECTION_SIGLIP,
    is_qdrant_healthy,
    query_collection,
)
from search.ocr_search import search_ocr
from search.asr_search import search_asr
from search.asr_metadata import get_asr_for_keyframe_metadata, preload_asr_metadata
from search.object_search import search_object
from search.fusion import reciprocal_rank_fusion
from config.rrf_config import (
    RRF_K_BY_SOURCE,
    DEFAULT_RRF_K,
    QDRANT_FETCH_MULTIPLIER,
    QDRANT_FETCH_MIN,
)
from dres.router import router as dres_router
from search.trake import trake_router
from search.temporal_search import temporal_router
from clipboard_router import router as clipboard_router

# ── Config ────────────────────────────────────────────────────────────────────
KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
METADATA_KEYFRAMES_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes"
GPU_DEVICE = "cuda:0"
USE_PECORE = False
USE_BGE_RERANKER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backend")

# ── App state ─────────────────────────────────────────────────────────────────
_models_loaded: bool = False
query_executor = ThreadPoolExecutor(max_workers=8)
# Executor riêng cho các tác vụ I/O nhẹ (đọc CSV metadata) — tách khỏi
# query_executor để không tranh chấp thread với Qdrant/OCR/ASR/Object.
io_executor = ThreadPoolExecutor(max_workers=8)


def _preload_all_keyframe_metadata() -> int:
    """
    Quét toàn bộ file *_keyframes.csv trong METADATA_KEYFRAMES_DIR và nạp
    trước vào cache RAM (qua get_video_keyframes) lúc server khởi động.

    Mục đích: loại bỏ hoàn toàn I/O đọc CSV trên NAS khỏi đường dẫn xử lý
    request — trước đây mỗi lần cache-miss theo video_id sẽ mở file ngay
    trong lúc build response, chạy đồng bộ trên event loop chính, gây chậm
    (và chặn toàn bộ server) dù không hề dùng OCR/ASR/Object.

    Trả về số video đã preload thành công.
    """
    if not os.path.isdir(METADATA_KEYFRAMES_DIR):
        logger.warning(
            "METADATA_KEYFRAMES_DIR không tồn tại (%s) — bỏ qua preload, "
            "sẽ load lazy theo từng request (có thể chậm lần đầu mỗi video).",
            METADATA_KEYFRAMES_DIR,
        )
        return 0

    t0 = time.perf_counter()
    loaded = 0
    total_files = 0
    try:
        fnames = [f for f in os.listdir(METADATA_KEYFRAMES_DIR) if f.endswith("_keyframes.csv")]
        total_files = len(fnames)
        logger.info("Tìm thấy %d file keyframe metadata cần preload từ NAS...", total_files)

        def _load_single_csv(fname):
            video_id = fname[: -len("_keyframes.csv")]
            try:
                kfs = get_video_keyframes(video_id)
                return 1 if kfs else 0
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(_load_single_csv, fnames))
            loaded = sum(results)
    except Exception:
        logger.exception("Lỗi khi liệt kê METADATA_KEYFRAMES_DIR")

    logger.info(
        "Preload keyframe metadata hoàn tất: %d/%d video thành công trong %.3f s",
        loaded,
        total_files,
        time.perf_counter() - t0,
    )
    return loaded


# ── Global RAM Image Cache ───────────────────────────────────────────────────
RAM_IMAGE_CACHE: Dict[str, bytes] = {}


def _preload_all_keyframe_images() -> int:
    """
    Nạp dữ liệu nhị phân (bytes) của các file ảnh keyframe từ KEYFRAME_ROOT vào RAM_IMAGE_CACHE.
    Nếu file nào nạp thất bại hoặc quá lớn, tự động bỏ qua để nạp lazy (fallback Disk) về sau.
    """
    if not os.path.isdir(KEYFRAME_ROOT):
        logger.warning("KEYFRAME_ROOT không tồn tại (%s) — bỏ qua RAM image preloading.", KEYFRAME_ROOT)
        return 0

    t0 = time.perf_counter()
    loaded = 0
    total_scanned = 0

    try:
        rel_paths = []
        for root, _, files in os.walk(KEYFRAME_ROOT):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    rel = os.path.relpath(os.path.join(root, f), KEYFRAME_ROOT)
                    rel_paths.append(rel)

        total_scanned = len(rel_paths)
        logger.info("Quét thấy %d file ảnh keyframe — bắt đầu preload vào RAM Server...", total_scanned)

        def _load_single_img(rel_path):
            abs_p = os.path.join(KEYFRAME_ROOT, rel_path)
            try:
                with open(abs_p, "rb") as fp:
                    RAM_IMAGE_CACHE[rel_path] = fp.read()
                return 1
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(_load_single_img, rel_paths))
            loaded = sum(results)

    except Exception:
        logger.exception("Lỗi khi quét/nạp keyframe images vào RAM")

    logger.info(
        "Preload RAM Image Cache hoàn tất: %d/%d ảnh nạp thành công vào RAM trong %.3f s",
        loaded,
        total_scanned,
        time.perf_counter() - t0,
    )
    return loaded


# ── Lifespan: load models once at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_loaded
    logger.info("=== Server starting — loading models & RAM caches ===")
    try:
        if USE_PECORE:
            load_pecore(device=GPU_DEVICE)
        load_siglip(device=GPU_DEVICE)
        if USE_BGE_RERANKER:
            load_bge_reranker(device=GPU_DEVICE)

        loop = asyncio.get_event_loop()
        # 1. Preload metadata keyframes vào RAM
        await loop.run_in_executor(io_executor, _preload_all_keyframe_metadata)

        # 2. Preload toàn bộ ASR metadata vào RAM
        await loop.run_in_executor(io_executor, preload_asr_metadata)

        # 3. Preload các file ảnh keyframe vào RAM Server
        await loop.run_in_executor(io_executor, _preload_all_keyframe_images)

        _models_loaded = True
        logger.info("=== Models & RAM Caches loaded ✓ ===")
    except Exception:
        logger.exception("Failed to load models / RAM caches")
    yield
    logger.info("=== Server shutting down ===")
    query_executor.shutdown(wait=False)
    io_executor.shutdown(wait=False)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AIC 2026 Search Backend",
    description="Text-to-image retrieval with PE-Core-L + SigLIP2 + RRF fusion",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS — allow React frontend (any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dres_router)
app.include_router(trake_router)
app.include_router(temporal_router)
app.include_router(clipboard_router)


# Aggressive caching for immutable keyframe images under /images/*.
@app.middleware("http")
async def cache_keyframe_images(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/images/"):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/images/{image_path:path}")
async def serve_keyframe_image(image_path: str):
    """
    Phục vụ ảnh keyframe trực tiếp từ RAM Server Cache.
    Nếu chưa có trong RAM (hoặc preload thiếu), tự động đọc từ Disk NAS (Fallback),
    ghi bổ sung vào RAM Cache cho lần sau và trả về cho Client.
    """
    clean_path = image_path.lstrip('/')

    # 1. RAM Cache HIT
    if clean_path in RAM_IMAGE_CACHE:
        return Response(
            content=RAM_IMAGE_CACHE[clean_path],
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-RAM-Cache": "HIT",
            },
        )

    # 2. Fallback to Disk NAS
    disk_path = os.path.join(KEYFRAME_ROOT, clean_path)
    if os.path.isfile(disk_path):
        try:
            with open(disk_path, "rb") as f:
                content = f.read()
            # Ghi bổ sung vào RAM Cache
            RAM_IMAGE_CACHE[clean_path] = content
            return Response(
                content=content,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "X-RAM-Cache": "MISS-FALLBACK-DISK",
                },
            )
        except Exception as e:
            logger.warning("Lỗi đọc file từ Disk NAS (%s): %s", disk_path, e)

    raise HTTPException(status_code=404, detail=f"Keyframe image '{image_path}' not found")

# Serve Frontend Web UI directly at root
FRONTEND_HTML = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Test_UI/index.html"))


@app.get("/")
async def serve_frontend():
    if os.path.exists(FRONTEND_HTML):
        return FileResponse(FRONTEND_HTML)
    raise HTTPException(status_code=404, detail="Frontend index.html not found")


# ── Helper: build image_url from image_path ───────────────────────────────────
def _image_url(request: Request, image_path: str) -> str:
    """
    Return a root-relative URL for a keyframe image.
    Using a relative path (/images/...) so it works correctly
    regardless of whether accessed directly or via SSH tunnel.
    image_path is the relative path stored in Qdrant payload
    (e.g. 'L24_V005/L24_V005_kf_0187.jpg').
    """
    return f"/images/{image_path}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Check server liveness, model status, and Qdrant connectivity."""
    return HealthResponse(
        status="ok",
        models_loaded=_models_loaded,
        qdrant_connected=is_qdrant_healthy(),
    )


@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search(req: SearchRequest, request: Request):
    """
    Multi-source search with optional OCR, ASR and Semantic inputs,
    fusing results using Reciprocal Rank Fusion (RRF).
    """
    if not _models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models are still loading. Retry in a moment.",
        )

    t_request_start = time.perf_counter()

    # 1. Parse and build search tasks from search_queries or legacy fields
    rrf_k = req.rrf_k if req.rrf_k is not None else DEFAULT_RRF_K
    qdrant_fetch_depth = max(req.top_k * QDRANT_FETCH_MULTIPLIER, QDRANT_FETCH_MIN)
    es_fetch_depth = max(req.top_k * 10, 100)
    video_list = req.video_list if req.video_list else None

    # 2. Define search executors
    loop = asyncio.get_event_loop()
    timings = {
        "embedding": 0.0,
        "qdrant": 0.0,
        "ocr": 0.0,
        "asr": 0.0,
        "object": 0.0,
        "image": 0.0,
        "reranking": 0.0,
        "metadata_loading": 0.0,
        "image_url_generation": 0.0,
        "dres_calls": 0.0,
    }

    async def run_semantic_query(q_text: str):
        t_emb_start = time.perf_counter()
        siglip_emb = await loop.run_in_executor(None, encode_text_siglip, q_text)
        timings["embedding"] += time.perf_counter() - t_emb_start

        t_qdrant_start = time.perf_counter()
        qdrant_hits = await loop.run_in_executor(
            query_executor,
            query_collection,
            siglip_emb,
            COLLECTION_SIGLIP,
            qdrant_fetch_depth,
            video_list
        )
        timings["qdrant"] += time.perf_counter() - t_qdrant_start

        return [
            {
                "video_id": hit.payload["video_id"],
                "keyframe_id": hit.payload["keyframe_id"],
                "frame_idx": hit.payload["frame_idx"],
                "image_path": _normalize_rel_image_path(
                    hit.payload.get("image_path", ""),
                    hit.payload.get("video_id", ""),
                    hit.payload.get("keyframe_id", ""),
                ),
                "score": float(hit.score)
            }
            for hit in qdrant_hits
        ]

    async def run_ocr_query(q_text: str):
        t_ocr_start = time.perf_counter()
        res = await loop.run_in_executor(
            query_executor,
            search_ocr,
            q_text,
            es_fetch_depth,
            video_list
        )
        timings["ocr"] += time.perf_counter() - t_ocr_start
        return res

    async def run_asr_query(q_text: str):
        t_asr_start = time.perf_counter()
        res = await loop.run_in_executor(
            query_executor,
            search_asr,
            q_text,
            es_fetch_depth,
            video_list
        )
        timings["asr"] += time.perf_counter() - t_asr_start
        return res

    async def run_object_query(q_text: str):
        t_obj_start = time.perf_counter()
        res = await loop.run_in_executor(
            query_executor,
            search_object,
            q_text,
            es_fetch_depth,
            video_list
        )
        timings["object"] += time.perf_counter() - t_obj_start
        return res

    async def run_image_query(raw_query: str, vid: str = "", fid: str = ""):
        t_img_start = time.perf_counter()
        raw_combined = raw_query.strip() if raw_query else (f"{vid}_{fid}" if (vid and fid) else (vid or fid)).strip()
        parsed_vid, parsed_fidx, parsed_kfid = parse_video_frame_id(raw_combined)
        if not parsed_vid and vid:
            parsed_vid = vid.strip()
        if parsed_fidx is None and fid and fid.strip().isdigit():
            parsed_fidx = int(fid.strip())

        clean_vid = parsed_vid.replace(".mp4", "").strip()
        if not clean_vid:
            return []

        # Find nearest keyframe using metadata
        nearest_kf = await loop.run_in_executor(
            io_executor,
            get_nearest_keyframe,
            clean_vid,
            parsed_fidx,
            parsed_kfid
        )

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
            logger.warning("Image query: cannot find image for video=%s frame=%r", clean_vid, parsed_fidx)
            return []

        # Load and encode image with SigLIP
        from PIL import Image
        try:
            with Image.open(img_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img_emb = await loop.run_in_executor(None, lambda: encode_images_siglip([rgb_img])[0])
        except Exception as e:
            logger.warning("Failed to encode image %s: %s", img_path, e)
            return []

        t_qdrant_start = time.perf_counter()
        qdrant_hits = await loop.run_in_executor(
            query_executor,
            query_collection,
            img_emb,
            COLLECTION_SIGLIP,
            qdrant_fetch_depth,
            video_list
        )
        timings["image"] += time.perf_counter() - t_img_start

        return [
            {
                "video_id": hit.payload["video_id"],
                "keyframe_id": hit.payload["keyframe_id"],
                "frame_idx": hit.payload["frame_idx"],
                "image_path": _normalize_rel_image_path(
                    hit.payload.get("image_path", ""),
                    hit.payload.get("video_id", ""),
                    hit.payload.get("keyframe_id", ""),
                ),
                "score": float(hit.score)
            }
            for hit in qdrant_hits
        ]

    # Build search tasks list
    search_tasks = []
    if req.search_queries:
        for idx, item in enumerate(req.search_queries):
            q_type = (item.type or "text").lower().strip()
            q_text = (item.text or item.query or "").strip()
            q_vid = (item.video_id or "").strip()
            q_fid = (item.frame_id or (str(item.frame_idx) if item.frame_idx is not None else "") or item.keyframe_id or "").strip()

            if q_type in ("image", "keyframe"):
                combined_img_str = q_text or (f"{q_vid}_{q_fid}" if (q_vid and q_fid) else (q_vid or q_fid))
                if combined_img_str:
                    s_name = f"image_{idx + 1}"
                    search_tasks.append((s_name, run_image_query(combined_img_str, q_vid, q_fid)))
            elif q_type == "ocr":
                if q_text:
                    s_name = f"ocr_{idx + 1}"
                    search_tasks.append((s_name, run_ocr_query(q_text)))
            elif q_type == "asr":
                if q_text:
                    s_name = f"asr_{idx + 1}"
                    search_tasks.append((s_name, run_asr_query(q_text)))
            elif q_type == "object":
                if q_text:
                    s_name = f"object_{idx + 1}"
                    search_tasks.append((s_name, run_object_query(q_text)))
            else: # text / semantic
                if q_text:
                    s_name = f"text_{idx + 1}"
                    search_tasks.append((s_name, run_semantic_query(q_text)))
    else:
        # Legacy single field mapping
        ocr_q = (req.ocr_query or "").strip()
        if not ocr_q and req.query and req.use_ocr:
            ocr_q = req.query.strip()
        asr_q = (req.asr_query or "").strip()
        if not asr_q and req.query and req.use_asr:
            asr_q = req.query.strip()
        semantic_q = (req.semantic_query or "").strip()
        if not semantic_q and req.query and not req.use_ocr and not req.use_asr:
            semantic_q = req.query.strip()
        object_q = (req.object_query or "").strip()

        if semantic_q:
            search_tasks.append(("semantic", run_semantic_query(semantic_q)))
        if ocr_q:
            search_tasks.append(("ocr", run_ocr_query(ocr_q)))
        if asr_q:
            search_tasks.append(("asr", run_asr_query(asr_q)))
        if object_q:
            search_tasks.append(("object", run_object_query(object_q)))

    if not search_tasks:
        return SearchResponse(
            query=req.query or "",
            top_k=req.top_k,
            rrf_k=rrf_k,
            results=[],
            sources_used=[]
        )

    # 3. Trigger active searches concurrently
    task_keys = [t[0] for t in search_tasks]
    coroutines = [t[1] for t in search_tasks]

    t_sources_start = time.perf_counter()
    try:
        results_list = await asyncio.gather(*coroutines)
    except Exception as exc:
        logger.exception("Concurrent query execution failed")
        raise HTTPException(status_code=500, detail=f"Query error: {exc}")
    sources_total = time.perf_counter() - t_sources_start

    results_by_source = dict(zip(task_keys, results_list))

    # 4. RRF Fusion / Reranking stage
    t_rerank_start = time.perf_counter()
    fetch_n = max(req.top_k, req.reranker_top_n) if req.use_reranker else req.top_k
    if len(results_by_source) == 1:
        source_name = task_keys[0]
        raw_results = results_by_source[source_name]
        sorted_results = sorted(raw_results, key=lambda x: x["score"], reverse=True)[:fetch_n]
        for item in sorted_results:
            item["matched_sources"] = [source_name]
        fused = sorted_results
    else:
        # Fuse multiple sources using RRF with adaptive k values per source
        logger.debug(
            "RRF fusion: sources=%s, top_k=%d, global_k=%d, per_source_k=%s",
            list(results_by_source.keys()),
            fetch_n,
            rrf_k,
            RRF_K_BY_SOURCE,
        )
        fused = reciprocal_rank_fusion(
            results_by_source=results_by_source,
            top_k=fetch_n,
            k=rrf_k,
            source_k=RRF_K_BY_SOURCE
        )

    # Determine query text for reranker
    first_text_query = ""
    query_repr_parts = []
    if req.search_queries:
        for sq in req.search_queries:
            q_type = (sq.type or "text").upper()
            t = (sq.text or sq.query or "").strip()
            if (sq.type or "").lower() in ("image", "keyframe"):
                img_desc = f"{sq.video_id}#{sq.frame_id}" if (sq.video_id and sq.frame_id) else (t or sq.video_id or "Image")
                query_repr_parts.append(f"Image({img_desc})")
            elif t:
                query_repr_parts.append(f"{q_type}: {t}")
                if not first_text_query:
                    first_text_query = t
    else:
        if semantic_q:
            query_repr_parts.append(f"Semantic: {semantic_q}")
            if not first_text_query:
                first_text_query = semantic_q
        if ocr_q:
            query_repr_parts.append(f"OCR: {ocr_q}")
            if not first_text_query:
                first_text_query = ocr_q
        if asr_q:
            query_repr_parts.append(f"ASR: {asr_q}")
            if not first_text_query:
                first_text_query = asr_q
        if object_q:
            query_repr_parts.append(f"Object: {object_q}")

    # Apply BGE-Reranker-v2-m3 if enabled
    if req.use_reranker and first_text_query:
        fused = await loop.run_in_executor(
            query_executor,
            rerank_candidates,
            first_text_query,
            fused,
            req.top_k,
            req.reranker_top_n,
        )
    else:
        fused = fused[:req.top_k]

    timings["reranking"] = time.perf_counter() - t_rerank_start

    # 5. Metadata loading stage (resolving timestamp per keyframe)
    t_meta_start = time.perf_counter()

    async def _resolve_metadata(item: dict):
        real_fidx, ts_ms, pts_s, rel_img_path = await loop.run_in_executor(
            io_executor,
            resolve_keyframe_metadata,
            item["video_id"],
            item.get("keyframe_id"),
            item.get("frame_idx"),
        )
        item["frame_idx"] = real_fidx
        item["pts_time"] = pts_s
        item["image_path"] = rel_img_path or item.get("image_path", "")
        return item, ts_ms, pts_s

    resolved = await asyncio.gather(*[_resolve_metadata(item) for item in fused])
    timings["metadata_loading"] = time.perf_counter() - t_meta_start

    # 6. Image URL generation & Response Object Building stage
    t_url_start = time.perf_counter()
    results = []
    for rank, (item, ts_ms, pts_s) in enumerate(resolved, start=1):
        results.append(
            ImageResult(
                rank=rank,
                score=item["score"],
                video_id=item["video_id"],
                keyframe_id=item["keyframe_id"],
                frame_idx=item["frame_idx"],
                pts_time=pts_s,
                timestamp=ts_ms,
                image_url=_image_url(request, item["image_path"]),
                matched_sources=item.get("matched_sources", [task_keys[0]] if len(task_keys) == 1 else [])
            )
        )
    timings["image_url_generation"] = time.perf_counter() - t_url_start

    total_time = time.perf_counter() - t_request_start

    # Comprehensive timing breakdown logging
    logger.info(
        "[TIMING] /api/search total=%.4fs | embedding=%.4fs | qdrant=%.4fs | rerank=%.4fs | metadata=%.4fs | url_gen=%.4fs | dres=%.4fs (OCR=%.4fs, ASR=%.4fs, OBJ=%.4fs, sources_wall=%.4fs)",
        total_time,
        timings["embedding"],
        timings["qdrant"],
        timings["reranking"],
        timings["metadata_loading"],
        timings["image_url_generation"],
        timings["dres_calls"],
        timings["ocr"],
        timings["asr"],
        timings["object"],
        sources_total,
    )

    query_str = req.query or (" | ".join(query_repr_parts) if query_repr_parts else "search")
    return SearchResponse(
        query=query_str,
        top_k=req.top_k,
        rrf_k=rrf_k,
        results=results,
        sources_used=task_keys
    )


@app.get("/api/video/{video_id}/keyframes", response_model=VideoKeyframesResponse, tags=["Video"])
async def get_keyframes_for_video(video_id: str):
    """
    Get all keyframes for a video sorted by frame index with timestamps and URLs.
    Cached in memory for instant response.
    """
    loop = asyncio.get_event_loop()
    kfs = await loop.run_in_executor(io_executor, get_video_keyframes, video_id)
    if not kfs:
        raise HTTPException(status_code=404, detail=f"No keyframes found for video '{video_id}'")
    # Include ASR text in the timeline response so the frontend can highlight
    # consecutive frames belonging to the same spoken segment without issuing
    # one request per thumbnail.
    for item in kfs:
        asr = get_asr_for_keyframe_metadata(
            video_id,
            frame_idx=item.get("frame_idx"),
            keyframe_id=item.get("keyframe_id"),
        ) or {}
        item["asr_6s"] = asr.get("asr_6s", "") or ""
        item["asr_text"] = asr.get("asr_text", "") or item["asr_6s"]
        item["full_text"] = asr.get("full_text", "") or item["asr_text"]

    return VideoKeyframesResponse(
        video_id=video_id,
        total_keyframes=len(kfs),
        fps=get_video_fps(video_id),
        keyframes=[KeyframeMeta(**k) for k in kfs],
    )


@app.get("/api/keyframe/asr", tags=["Video"])
async def get_keyframe_asr(video_id: str, frame_idx: int, keyframe_id: Optional[str] = None):
    """
    Get the ASR segment for a specific keyframe from local metadata
    at /dev/shm/asr (JSONL files),
    returning both the 6s-windowed transcript and the full transcript.
    """
    loop = asyncio.get_event_loop()
    asr_result = await loop.run_in_executor(
        io_executor,
        get_asr_for_keyframe_metadata,
        video_id,
        frame_idx,
        keyframe_id
    )
    asr_6s = ""
    full_text = ""
    asr_text = ""
    if isinstance(asr_result, dict):
        asr_6s = asr_result.get("asr_6s", "") or asr_result.get("asr_text", "") or ""
        full_text = asr_result.get("full_text", "") or asr_6s or ""
        asr_text = asr_6s or full_text or ""
    return {
        "asr_text": asr_text or "",
        "asr_6s": asr_6s or "",
        "full_text": full_text or "",
    }



# ── Check Video: Frame Timestamp Lookup ───────────────────────────────────────
@app.get("/api/v1/frame/timestamp", tags=["Video"])
@app.get("/api/frame/timestamp", tags=["Video"])
async def get_frame_timestamp_endpoint(
    video_id: str,
    frame_idx: Optional[int] = None,
    keyframe_id: Optional[str] = None,
):
    """
    Get timestamp, pts_time, fps, and keyframe info for a given video_id and frame_idx (or keyframe_id).
    Validates if video and frame exist in metadata.
    Uses exact FPS from videos_metadata.csv.
    """
    clean_video_id = video_id.replace(".mp4", "").strip()
    if not clean_video_id:
        raise HTTPException(status_code=400, detail="video_id không được để trống")

    loop = asyncio.get_event_loop()
    kfs = await loop.run_in_executor(io_executor, get_video_keyframes, clean_video_id)
    if not kfs:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy video '{clean_video_id}' trong dataset")

    # Match by keyframe_id or frame_idx
    target_kf = None
    if keyframe_id:
        clean_kfid = keyframe_id.strip()
        for k in kfs:
            if k.get("keyframe_id") == clean_kfid:
                target_kf = k
                break

    if target_kf is None and frame_idx is not None:
        for k in kfs:
            if k.get("frame_idx") == frame_idx:
                target_kf = k
                break

    # Check if keyframe_id is numeric (e.g. "46")
    if target_kf is None and keyframe_id:
        try:
            numeric_idx = int(keyframe_id.strip())
            for k in kfs:
                if k.get("frame_idx") == numeric_idx:
                    target_kf = k
                    break
        except ValueError:
            pass

    # Fallback: find nearest keyframe by frame_idx (for sparse keyframe datasets)
    # where frame numbers skip (e.g. 0, 8, 24, 39...) and user entered a number in between.
    if target_kf is None and kfs:
        lookup_idx = None
        if frame_idx is not None:
            lookup_idx = frame_idx
        elif keyframe_id:
            try:
                lookup_idx = int(keyframe_id.strip())
            except ValueError:
                pass
        if lookup_idx is not None:
            target_kf = min(kfs, key=lambda k: abs(k.get("frame_idx", 0) - lookup_idx))

    if target_kf is None:
        requested_identifier = keyframe_id if keyframe_id is not None else frame_idx
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy frame '{requested_identifier}' trong video '{clean_video_id}' (video có {len(kfs)} frames)"
        )

    ts_ms = target_kf.get("timestamp", 0)
    pts_s = target_kf.get("pts_time", round(ts_ms / 1000.0, 3))
    v_fps = get_video_fps(clean_video_id)
    v_info = get_video_info(clean_video_id)

    return {
        "video_id": clean_video_id,
        "keyframe_id": target_kf.get("keyframe_id", ""),
        "frame_idx": target_kf.get("frame_idx", frame_idx if frame_idx is not None else 0),
        "timestamp": ts_ms,
        "pts_time": pts_s,
        "fps": v_fps,
        "video_info": v_info,
        "image_url": target_kf.get("image_url", ""),
        "image_path": target_kf.get("image_path", ""),
        "total_keyframes": len(kfs),
        "video_stream_url": f"/api/video/{clean_video_id}/stream",
    }


# ── Video Metadata & FPS Endpoints ───────────────────────────────────────────
@app.get("/api/videos/fps", tags=["Video"])
async def get_all_videos_fps_endpoint():
    """Return dictionary of all {video_id: fps} from videos_metadata.csv."""
    return {"videos_fps": get_all_videos_fps()}


@app.get("/api/video/{video_id}/fps", tags=["Video"])
async def get_single_video_fps_endpoint(video_id: str):
    """Return exact FPS and metadata for a specific video."""
    clean_vid = video_id.replace(".mp4", "").strip()
    return {
        "video_id": clean_vid,
        "fps": get_video_fps(clean_vid),
        "info": get_video_info(clean_vid),
    }


# ── Check Video: MP4 Stream ────────────────────────────────────────────────────
@app.get("/api/video/{video_id}/stream", tags=["Video"])
@app.get("/api/v1/video/{video_id}/stream", tags=["Video"])
async def stream_video(video_id: str):
    """Stream MP4 video file directly for browser playback."""
    clean_vid = video_id.replace(".mp4", "").strip()
    video_path = find_video_path(clean_vid)
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file video MP4 cho '{clean_vid}'")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{clean_vid}.mp4")


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=False)
