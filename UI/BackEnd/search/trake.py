import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 1. Models & Encoders
from models.siglip_encoder import encode_text_siglip

# 2. Search Engines & Utils
from search.qdrant_search import query_collection, COLLECTION_SIGLIP
from search.ocr_search import search_ocr
from search.asr_search import search_asr
from search.fusion import reciprocal_rank_fusion
from search.keyframe_manager import get_keyframe_timestamp, _normalize_rel_image_path

logger = logging.getLogger(__name__)

# Khoảng cách tối thiểu (ms) để coi là một hành động khác biệt thay vì frame lân cận của cùng 1 hành động
MIN_DISTINCT_EVENT_DIFF_MS = 10000  # 10 giây

def _format_time_sec(ts_ms: int) -> str:
    total_sec = max(0, ts_ms // 1000)
    m = total_sec // 60
    s = total_sec % 60
    return f"{m:02d}:{s:02d}"

# ==========================================
# 1. ĐỊNH NGHĨA PAYLOAD API 
# ==========================================
class TrakeEvent(BaseModel):
    event_id: str
    description: Optional[str] = None
    text_query: Optional[str] = None
    ocr_query: Optional[str] = None
    asr_query: Optional[str] = None
    weight_text: float = 1.0
    weight_ocr: float = 1.0
    weight_asr: float = 1.0
    use_semantic: Optional[bool] = None
    use_ocr: Optional[bool] = None
    use_asr: Optional[bool] = None

class TrakeChallengeRequest(BaseModel):
    global_context: Optional[str] = None
    events: List[TrakeEvent]
    top_k_videos: int = 20
    video_list: Optional[List[str]] = None
    use_ocr: bool = False
    use_asr: bool = False

class TrakeEventResult(BaseModel):
    event_id: str
    keyframe_id: str
    frame_idx: int
    timestamp: int
    image_path: str
    score: float
    matched_sources: List[str] = []

class TrakeVideoResult(BaseModel):
    video_id: str
    sequence_id: int = 1
    time_span: Optional[str] = None
    duration_sec: Optional[float] = None
    global_score: float
    timeline: List[TrakeEventResult]

class TrakeChallengeResponse(BaseModel):
    status: str
    message: str
    data: List[TrakeVideoResult]

# ==========================================
# 2. TRAKE PROCESSOR (Beam Search DP + Action-Distance NMS + SigLIP2 + RRF Fusion)
# ==========================================
class TrakeAdvancedProcessor:
    def __init__(self, max_workers: int = 32):
        logger.info("✅ TRAKE Beam Search DP-Engine đã sẵn sàng với Semantic + [OCR, ASR]!")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _run_semantic(self, text: str, limit: int, video_list: Optional[List[str]] = None) -> List[dict]:
        """Luồng tìm kiếm Qdrant (Semantic)"""
        try:
            emb = encode_text_siglip(text)
            pts = query_collection(embedding=emb, collection_name=COLLECTION_SIGLIP, limit=limit, video_list=video_list)
            return [
                {
                    "video_id": pt.payload.get("video_id"),
                    "keyframe_id": pt.payload.get("keyframe_id"),
                    "frame_idx": int(pt.payload.get("frame_idx", 0)),
                    "timestamp": pt.payload.get("timestamp"),
                    "image_path": pt.payload.get("image_path"),
                    "score": float(pt.score)
                }
                for pt in pts
                if pt.payload and pt.payload.get("video_id")
            ]
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

    def _search_single_engine(self, engine_name: str, text: str, limit: int, video_list: Optional[List[str]]) -> Tuple[str, List[dict]]:
        """Chạy một engine tìm kiếm đơn lẻ"""
        try:
            if not text or not text.strip():
                return engine_name, []
            if engine_name == "semantic":
                res = self._run_semantic(text.strip(), limit, video_list)
            elif engine_name == "ocr":
                res = search_ocr(text.strip(), top_k=limit, video_list=video_list)
            elif engine_name == "asr":
                res = search_asr(text.strip(), top_k=limit, video_list=video_list)
            else:
                res = []
            return engine_name, res
        except Exception as e:
            logger.error(f"Engine {engine_name} error with query '{text}': {e}")
            return engine_name, []

    def _fuse_engine_results(self, engine_results: Dict[str, List[dict]], limit: int = 150) -> List[dict]:
        """Dung hợp kết quả từ các engine bằng Reciprocal Rank Fusion (RRF)"""
        valid_runs = []
        run_names = []
        for name, run in engine_results.items():
            if run:
                valid_runs.append(run)
                run_names.append(name)

        if not valid_runs:
            return []
        if len(valid_runs) == 1:
            res = valid_runs[0]
            for pt in res:
                pt["matched_sources"] = [run_names[0]]
            return res[:limit]

        return reciprocal_rank_fusion(valid_runs, run_names=run_names, limit=limit)

    def process(self, request: TrakeChallengeRequest) -> List[Dict]:
        start_time = time.time()
        num_events = len(request.events)
        logger.info(f"[TRAKE] Bắt đầu giải bài toán chuỗi thời gian ({num_events} sự kiện)")

        fetch_limit = max(100, request.top_k_videos * 8)

        # BƯỚC 1: Lấy các tasks tìm kiếm song song cho từng event
        futures_map = {}
        for event in request.events:
            e_text = event.text_query or event.description or ""
            use_sem = event.use_semantic if event.use_semantic is not None else True
            use_o = event.use_ocr if event.use_ocr is not None else request.use_ocr
            use_a = event.use_asr if event.use_asr is not None else request.use_asr

            if use_sem and e_text.strip():
                f = self.executor.submit(self._search_single_engine, "semantic", e_text, fetch_limit, request.video_list)
                futures_map[f] = (event.event_id, "semantic")

            o_text = event.ocr_query or (e_text if use_o else "")
            if use_o and o_text.strip():
                f = self.executor.submit(self._search_single_engine, "ocr", o_text, fetch_limit, request.video_list)
                futures_map[f] = (event.event_id, "ocr")

            a_text = event.asr_query or (e_text if use_a else "")
            if use_a and a_text.strip():
                f = self.executor.submit(self._search_single_engine, "asr", a_text, fetch_limit, request.video_list)
                futures_map[f] = (event.event_id, "asr")

        # Task tìm bối cảnh toàn cục (nếu có)
        has_context = bool(request.global_context and request.global_context.strip())
        if has_context:
            ctx_text = request.global_context.strip()
            f = self.executor.submit(self._search_single_engine, "semantic", ctx_text, fetch_limit, request.video_list)
            futures_map[f] = ("__GLOBAL_CONTEXT__", "semantic")
            if request.use_ocr:
                f = self.executor.submit(self._search_single_engine, "ocr", ctx_text, fetch_limit, request.video_list)
                futures_map[f] = ("__GLOBAL_CONTEXT__", "ocr")
            if request.use_asr:
                f = self.executor.submit(self._search_single_engine, "asr", ctx_text, fetch_limit, request.video_list)
                futures_map[f] = ("__GLOBAL_CONTEXT__", "asr")

        # BƯỚC 2: Thu thập kết quả
        grouped_engine_results: Dict[str, Dict[str, List[dict]]] = {}
        for event in request.events:
            grouped_engine_results[event.event_id] = {}
        context_engine_results: Dict[str, List[dict]] = {}

        for fut in as_completed(futures_map):
            target_id, engine_name = futures_map[fut]
            try:
                _, res = fut.result()
                if target_id == "__GLOBAL_CONTEXT__":
                    context_engine_results[engine_name] = res
                else:
                    grouped_engine_results[target_id][engine_name] = res
            except Exception as e:
                logger.error(f"[TRAKE] Lỗi task ({target_id}, {engine_name}): {e}")

        # BƯỚC 3: RRF Fusion cho từng sự kiện
        all_event_candidates: Dict[str, List[dict]] = {}
        potential_videos = set()

        for event in request.events:
            fused_pts = self._fuse_engine_results(grouped_engine_results[event.event_id], limit=fetch_limit)
            event_cands = []
            seen_kf = set()

            for pt in fused_pts:
                vid = pt.get("video_id")
                if not vid:
                    continue
                kfid = pt.get("keyframe_id")
                fidx = int(pt.get("frame_idx", 0))

                # Deduplicate cùng video và cùng keyframe trong 1 event
                dedup_key = (vid, kfid or fidx)
                if dedup_key in seen_kf:
                    continue
                seen_kf.add(dedup_key)

                ts = pt.get("timestamp")
                if ts is None:
                    ts = get_keyframe_timestamp(vid, kfid, fidx)

                event_cands.append({
                    "event_id": event.event_id,
                    "video_id": vid,
                    "keyframe_id": kfid or "",
                    "frame_idx": fidx,
                    "timestamp": int(ts),
                    "image_path": _normalize_rel_image_path(pt.get("image_path") or "", vid, kfid or f"{vid}_kf_{fidx:04d}"),
                    "score": float(pt.get("score", 0.0)),
                    "matched_sources": pt.get("matched_sources", [])
                })
                potential_videos.add(vid)

            all_event_candidates[event.event_id] = event_cands

        if not potential_videos:
            logger.warning("[TRAKE] Không tìm thấy bất kỳ manh mối nào từ các sự kiện.")
            return []

        # Điểm bối cảnh chung
        context_bonus_scores: Dict[str, float] = {}
        if has_context and context_engine_results:
            ctx_fused = self._fuse_engine_results(context_engine_results, limit=fetch_limit)
            for pt in ctx_fused:
                vid = pt.get("video_id")
                if vid:
                    context_bonus_scores[vid] = max(
                        context_bonus_scores.get(vid, 0.0),
                        float(pt.get("score", 0.0))
                    )

        # BƯỚC 4: Gom candidate theo từng video & Early Pruning
        candidates_by_video: Dict[str, Dict[str, List[dict]]] = {
            vid: {e.event_id: [] for e in request.events} for vid in potential_videos
        }
        for eid, frames in all_event_candidates.items():
            for frame in frames:
                candidates_by_video[frame["video_id"]][eid].append(frame)

        # Lọc ngay các video KHÔNG chứa đủ tất cả các sự kiện
        valid_videos = [
            vid for vid in potential_videos
            if all(len(candidates_by_video[vid][e.event_id]) > 0 for e in request.events)
        ]

        logger.info(f"[TRAKE] Tìm thấy {len(potential_videos)} video tiềm năng, {len(valid_videos)} video có đủ {num_events} sự kiện.")

        # BƯỚC 5: Beam Search DP + Action-Distance NMS cho từng video hợp lệ
        final_results = []
        for vid in valid_videos:
            sequences = self._beam_search_dp_alignment(
                vid, request.events, candidates_by_video[vid], max_sequences=3, beam_width=6
            )
            for seq in sequences:
                # Cộng điểm bối cảnh (15% bonus)
                bonus = context_bonus_scores.get(vid, 0.0) * 0.15
                seq["global_score"] = round(seq["global_score"] + bonus, 4)
                final_results.append(seq)

        final_results.sort(key=lambda x: x["global_score"], reverse=True)
        elapsed = time.time() - start_time
        logger.info(f"[TRAKE] Hoàn thành xử lý trong {elapsed:.2f}s, trả về {len(final_results[:request.top_k_videos])} chuỗi kết quả (từ {len(valid_videos)} video).")
        return final_results[:request.top_k_videos]

    def _beam_search_dp_alignment(
        self,
        vid: str,
        events: List[TrakeEvent],
        video_cands: Dict[str, List[Dict]],
        max_sequences: int = 3,
        beam_width: int = 6,
        min_score_ratio: float = 0.55
    ) -> List[Dict]:
        """
        Beam Search DP giữ lại Top-B đường đi tại mỗi trạng thái, không xóa frame.
        Sau đó áp dụng Action-Distance NMS để chọn tối đa Top-K chuỗi thực sự khác biệt
        (khác ít nhất 1 frame và frame khác đó phải cách ít nhất MIN_DISTINCT_EVENT_DIFF_MS).
        """
        n_events = len(events)
        if n_events == 0:
            return []

        # Sắp xếp candidate của từng event theo timestamp tăng dần
        sorted_cands = {}
        for e in events:
            sorted_cands[e.event_id] = sorted(video_cands.get(e.event_id, []), key=lambda x: x["timestamp"])

        e0_id = events[0].event_id
        if not sorted_cands[e0_id]:
            return []

        # dp[i] là một dict mapping: cand_idx -> danh sách Top-B paths dẫn tới candidate đó
        # Một path được biểu diễn bằng: {'score': float, 'timeline': List[cand]}
        dp: List[List[Dict[str, Any]]] = []

        # Khởi tạo trạng thái Event 0: mỗi candidate có 1 path ban đầu
        init_layer = []
        for cand in sorted_cands[e0_id]:
            init_layer.append({
                'cand': cand,
                'paths': [{
                    'score': cand['score'],
                    'timeline': [cand]
                }]
            })
        dp.append(init_layer)

        # Chuyển trạng thái từ Event i-1 sang Event i bằng Beam Search
        for i in range(1, n_events):
            e_curr = events[i].event_id
            curr_cands = sorted_cands[e_curr]
            if not curr_cands:
                return []

            prev_layer = dp[i - 1]
            curr_layer: List[Dict[str, Any]] = []

            for cand_curr in curr_cands:
                curr_ts = cand_curr["timestamp"]
                incoming_paths = []

                for prev_entry in prev_layer:
                    prev_cand = prev_entry['cand']
                    prev_ts = prev_cand['timestamp']
                    if prev_ts >= curr_ts:
                        break  # vì prev_layer được sắp xếp theo timestamp

                    # Tính Temporal Distance Penalty
                    dt_sec = (curr_ts - prev_ts) / 1000.0
                    penalty = min(0.20, 0.00012 * max(0.0, dt_sec - 30.0)) if dt_sec > 30.0 else 0.0

                    for p in prev_entry['paths']:
                        if p['score'] < 0:
                            continue
                        new_score = p['score'] - penalty + cand_curr['score']
                        incoming_paths.append({
                            'score': new_score,
                            'timeline': p['timeline'] + [cand_curr]
                        })

                # Sắp xếp và giữ lại Top-B đường đi tốt nhất tới candidate cand_curr
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

        # Thu thập toàn bộ các candidate paths tại bước cuối cùng (Event N-1)
        all_final_paths = []
        for entry in dp[-1]:
            all_final_paths.extend(entry['paths'])

        if not all_final_paths:
            return []

        # Sắp xếp tất cả các paths theo điểm tổng giảm dần
        all_final_paths.sort(key=lambda x: x['score'], reverse=True)

        # Áp dụng Action-Distance NMS:
        # Chuỗi Sk được chấp nhận nếu:
        # 1. Điểm >= 0.55 * Điểm chuỗi tốt nhất S1
        # 2. Khác với tất cả các chuỗi đã chọn trước đó ở ít nhất 1 mốc sự kiện
        #    và mốc khác đó phải cách ít nhất MIN_DISTINCT_EVENT_DIFF_MS (10s)
        selected_sequences: List[Dict] = []
        best_score = all_final_paths[0]['score']

        for path_entry in all_final_paths:
            p_score = path_entry['score']
            if p_score < best_score * min_score_ratio or p_score <= 0:
                break

            tl = path_entry['timeline']

            # Kiểm tra tính khác biệt hành động so với các chuỗi đã chọn
            is_distinct = True
            for sel in selected_sequences:
                sel_tl = sel['raw_timeline']
                has_distinct_event = False
                for k in range(n_events):
                    dt = abs(tl[k]['timestamp'] - sel_tl[k]['timestamp'])
                    if dt >= MIN_DISTINCT_EVENT_DIFF_MS:
                        has_distinct_event = True
                        break
                if not has_distinct_event:
                    # Chuỗi này quá gần (trùng hoặc chỉ là frame rung lắc lân cận của cùng 1 hành động)
                    is_distinct = False
                    break

            if not is_distinct:
                continue

            # Tạo định dạng output chuẩn
            clean_timeline = []
            for cand in tl:
                clean_timeline.append({
                    "event_id": cand["event_id"],
                    "keyframe_id": cand["keyframe_id"],
                    "frame_idx": cand["frame_idx"],
                    "timestamp": cand["timestamp"],
                    "image_path": _normalize_rel_image_path(cand.get("image_path") or "", cand.get("video_id") or vid, cand.get("keyframe_id") or f"{vid}_kf_{cand['frame_idx']:04d}"),
                    "score": round(cand["score"], 6),
                    "matched_sources": cand.get("matched_sources", [])
                })

            t_start = clean_timeline[0]["timestamp"]
            t_end = clean_timeline[-1]["timestamp"]
            duration_sec = round((t_end - t_start) / 1000.0, 1)
            time_span = f"{_format_time_sec(t_start)} -> {_format_time_sec(t_end)}"

            seq_id = len(selected_sequences) + 1
            selected_sequences.append({
                "video_id": vid,
                "sequence_id": seq_id,
                "time_span": time_span,
                "duration_sec": duration_sec,
                "global_score": round(max(0.0, p_score / n_events), 4),
                "timeline": clean_timeline,
                "raw_timeline": tl
            })

            if len(selected_sequences) >= max_sequences:
                break

        # Loại bỏ trường phụ 'raw_timeline' trước khi trả về
        for s in selected_sequences:
            s.pop("raw_timeline", None)

        return selected_sequences

# ==========================================
# 3. ĐĂNG KÝ ROUTER API
# ==========================================
trake_advanced_processor = None
trake_router = APIRouter(prefix="/api/v1/trake", tags=["TRAKE Challenge"])

@trake_router.post("/solve", response_model=TrakeChallengeResponse)
def solve_trake_endpoint(request: TrakeChallengeRequest):
    global trake_advanced_processor
    if trake_advanced_processor is None:
        trake_advanced_processor = TrakeAdvancedProcessor()
        
    try:
        results = trake_advanced_processor.process(request)
        return TrakeChallengeResponse(
            status="success",
            message="Đã trích xuất chuỗi thời gian thành công",
            data=results
        )
    except Exception as e:
        logger.exception("Lỗi khi giải đề TRAKE:")
        raise HTTPException(status_code=500, detail=str(e))