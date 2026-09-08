from typing import Dict, List, Optional, Union

DEFAULT_K = 60  # Giá trị k chuẩn theo công thức RRF gốc của Cormack et al.


def reciprocal_rank_fusion(
    search_runs: Optional[List[dict]] = None,
    top_k: int = 50,
    k: int = DEFAULT_K,
    results_by_source: Optional[Dict[str, List[dict]]] = None,
    source_k: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Merge kết quả từ nhiều lượt search (kể cả cùng source_type) bằng RRF chuẩn.

    Args:
        search_runs: Danh sách các lượt search. Mỗi element có dạng:
                     {
                         "source_name": "text_1",  # Hoặc "semantic", "ocr", "asr", "image_1"...
                         "results": [ {"keyframe_id": "...", ...}, ... ]
                     }
        top_k: Số lượng kết quả hợp nhất trả về.
        k: Hằng số RRF damping (mặc định = 60).
        results_by_source: Dict dạng { "semantic": [...], "ocr": [...] } (tương thích ngược).
        source_k: Dict mapping từng source_name sang giá trị damping k riêng biệt (optional).

    Returns:
        Danh sách các dict đã được cộng dồn điểm RRF và sắp xếp giảm dần theo score.
    """
    if k <= 0:
        raise ValueError(f"Hằng số k phải > 0, nhận được k={k}")

    # Chuẩn hoá input thành danh sách search_runs
    runs: List[dict] = []
    if search_runs is not None:
        runs.extend(search_runs)
    elif results_by_source is not None:
        for s_name, res in results_by_source.items():
            runs.append({"source_name": s_name, "results": res})

    if not runs:
        return []

    scores: Dict[str, float] = {}
    doc_data: Dict[str, dict] = {}

    for run_idx, run in enumerate(runs):
        source_name = run.get("source_name", f"source_{run_idx + 1}")
        results = run.get("results", [])

        # Xác định k cho nguồn này (nếu có trong source_k)
        curr_k = k
        if source_k and source_name in source_k:
            curr_k = source_k[source_name]
        elif source_k:
            # Match prefix (ví dụ: 'text_1' -> 'text' / 'semantic')
            for prefix, pk in source_k.items():
                if source_name.startswith(prefix):
                    curr_k = pk
                    break

        seen_in_run = set()  # Dedup trong cùng 1 lượt search
        rank = 0

        for doc in results:
            key = doc.get("keyframe_id")
            if key is None:
                continue

            if key in seen_in_run:
                continue
            seen_in_run.add(key)

            rank += 1  # Rank bắt đầu từ 1

            # Công thức RRF chuẩn: 1 / (k + rank)
            rrf_score = 1.0 / (curr_k + rank)
            scores[key] = scores.get(key, 0.0) + rrf_score

            # Lưu thông tin document (giữ bản ghi đầu tiên xuất hiện)
            if key not in doc_data:
                doc_data[key] = dict(doc)

            # Đánh dấu các lượt search/nguồn mà doc này match được
            if "matched_sources" not in doc_data[key]:
                doc_data[key]["matched_sources"] = []
            if source_name not in doc_data[key]["matched_sources"]:
                doc_data[key]["matched_sources"].append(source_name)

    # Tie-break bằng keyframe_id để đảm bảo thứ tự kết quả ổn định (deterministic)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    final_results = []
    for key, score in ranked[:top_k]:
        d = doc_data[key]
        d["score"] = round(score, 6)
        final_results.append(d)

    return final_results