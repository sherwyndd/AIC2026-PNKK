import logging
import requests
from typing import List, Optional

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

def search_ocr(query: str, limit: int, video_list: Optional[List[str]] = None) -> List[dict]:
    """
    Search keyframes using OCR text from Elasticsearch.
    Uses BM25 + Fuzzy matching configuration.
    """
    if not query:
        return []
    
    url = f"{ES_HOST}/{INDEX_NAME}/_search"
    
    # Match query with fuzzy AUTO
    match_query = {
        "ocr_text": {
            "query": query,
            "fuzziness": "AUTO",
            "prefix_length": 1,
            "operator": "or",
            "minimum_should_match": "70%"
        }
    }
    
    # If video filter is provided
    if video_list:
        es_query = {
            "bool": {
                "must": [
                    {"match": match_query}
                ],
                "filter": [
                    {"terms": {"video_id.keyword": video_list}}
                ]
            }
        }
    else:
        es_query = {"match": match_query}
        
    payload = {
        "query": es_query,
        "size": int(limit),
        "min_score": 1.0,
        "_source": ["video_id", "keyframe_id", "frame_idx", "image_path", "ocr_text"]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error("Elasticsearch OCR search failed with status %d: %s", response.status_code, response.text)
            return []
            
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            
            # Normalize image_path (strip legacy absolute dataset prefixes)
            img_path = _normalize_img_path(source.get("image_path", ""))
                
            results.append({
                "video_id": source.get("video_id", ""),
                "keyframe_id": source.get("keyframe_id", ""),
                "frame_idx": source.get("frame_idx", 0),
                "image_path": img_path,
                "score": float(score),
                "ocr_text": source.get("ocr_text", "")
            })
        return results
    except Exception as exc:
        logger.exception("Error searching OCR in Elasticsearch: %s", exc)
        return []
