import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ES_HOST = "http://127.0.0.1:9200"
INDEX_NAME = "aic2026_elastics_text"

_OLD_PATH_PREFIXES = (
    "/AIClub_NAS/core_baotg/nhan/dataset/keyframes/",
    "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes/",
    "/AIClub_NAS/core_baotg/nhan/dataset/",
    "/mlcv1/WorkingSpace/Personal/baotg/AIC_2026/dataset/keyframes/",
    "/mlcv1/WorkingSpace/Personal/baotg/AIC_2026/dataset/",
    "/mlcv1/WorkingSpace/Personal/baotg/AIC_2026/keyframes/",
    "/workspace/dataset/keyframes/",
    "/workspace/dataset/metadata/keyframes/",
    "/workspace/dataset/",
    "/keyframes/",
    "keyframes/",
    "/dataset/keyframes/",
    "/dataset/",
)


def _normalize_img_path(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).replace("\\", "/").strip()
    for prefix in _OLD_PATH_PREFIXES:
        while s.startswith(prefix):
            s = s[len(prefix):]
    while s.startswith("/"):
        s = s[1:]
    while "//" in s:
        s = s.replace("//", "/")
    return s

_LABEL_ALIASES = {
    "traffic light": "traffic_light",
    "traffic-light": "traffic_light",
    "stop sign": "stop_sign",
    "stop-sign": "stop_sign",
    "fire hydrant": "fire_hydrant",
    "fire-hydrant": "fire_hydrant",
    "parking meter": "parking_meter",
    "parking-meter": "parking_meter",
    "dining table": "dining_table",
    "dining-table": "dining_table",
    "sports ball": "sports_ball",
    "sports-ball": "sports_ball",
    "tennis racket": "tennis_racket",
    "tennis-racket": "tennis_racket",
    "baseball bat": "baseball_bat",
    "baseball-bat": "baseball_bat",
    "baseball glove": "baseball_glove",
    "baseball-glove": "baseball_glove",
    "wine glass": "wine_glass",
    "wine-glass": "wine_glass",
    "hot dog": "hot_dog",
    "hot-dog": "hot_dog",
    "cell phone": "cell_phone",
    "cell-phone": "cell_phone",
    "hair drier": "hair_drier",
    "hair-drier": "hair_drier",
    "potted plant": "potted_plant",
    "potted-plant": "potted_plant",
}


def _normalize_label(label: str) -> str:
    cleaned = label.strip().lower().replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _LABEL_ALIASES.get(cleaned, cleaned.replace(" ", "_"))


def parse_object_query(query: str) -> List[Tuple[str, int]]:
    text = query.strip().lower()
    if not text:
        return []

    tokens = text.split()
    has_count_prefix = any(token.isdigit() for token in tokens)

    if has_count_prefix:
        parsed: List[Tuple[str, int]] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.isdigit():
                count = int(token)
                idx += 1
                if idx >= len(tokens):
                    break
                label_parts = [tokens[idx]]
                idx += 1
                while idx < len(tokens) and not tokens[idx].isdigit():
                    label_parts.append(tokens[idx])
                    idx += 1
                label = _normalize_label(" ".join(label_parts))
                if count > 0:
                    parsed.append((label, count))
            else:
                label_parts = [token]
                idx += 1
                while idx < len(tokens) and not tokens[idx].isdigit():
                    label_parts.append(tokens[idx])
                    idx += 1
                parsed.append((_normalize_label(" ".join(label_parts)), 1))
        return parsed

    normalized_tokens = [_normalize_label(token) for token in tokens]
    return list(Counter(normalized_tokens).items())


def expand_object_query(query: str) -> str:
    parts: List[str] = []
    for label, count in parse_object_query(query):
        parts.extend([label] * count)
    return " ".join(parts)


def _exact_count_clause(label: str, count: int) -> dict:
    """Match keyframe có ĐÚNG `count` lần label này (dùng obj_counts flattened field)."""
    return {
        "term": {
            f"obj_counts.{label}": str(count)
        }
    }


def search_object(query: str, limit: int, video_list: Optional[List[str]] = None) -> List[dict]:
    """
    Search keyframes theo đúng số lượng object detect được (exact match),
    dùng field obj_counts (flattened) thay vì object_text intervals.
    """
    parsed = parse_object_query(query)
    if not parsed:
        return []

    # gộp các entry trùng label (vd parse ra [("person",1),("car",2),("person",1)])
    merged = Counter()
    for label, count in parsed:
        merged[label] += count

    expanded = expand_object_query(query)
    url = f"{ES_HOST}/{INDEX_NAME}/_search"

    must_clauses = [_exact_count_clause(label, count) for label, count in merged.items()]
    if video_list:
        es_query = {
            "bool": {
                "must": must_clauses,
                "filter": [{"terms": {"video_id.keyword": video_list}}],
            }
        }
    else:
        es_query = {"bool": {"must": must_clauses}}

    payload = {
        "query": es_query,
        "size": int(limit),
        "_source": ["video_id", "keyframe_id", "frame_idx", "image_path", "object_text"],
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error(
                "Elasticsearch object search failed with status %d: %s",
                response.status_code,
                response.text,
            )
            return []

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])

        results = []
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)

            img_path = _normalize_img_path(source.get("image_path", ""))

            results.append(
                {
                    "video_id": source.get("video_id", ""),
                    "keyframe_id": source.get("keyframe_id", ""),
                    "frame_idx": source.get("frame_idx", 0),
                    "image_path": img_path,
                    "score": float(score),
                    "object_text": source.get("object_text", ""),
                    "object_query_expanded": expanded,
                }
            )
        return results
    except Exception as exc:
        logger.exception("Error searching objects in Elasticsearch: %s", exc)
        return []