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

def search_asr(query: str, limit: int, video_list: Optional[List[str]] = None) -> List[dict]:
    """
    Search keyframes using ASR text from Elasticsearch.
    Uses BM25 matching (no fuzziness).
    """
    if not query:
        return []
    
    url = f"{ES_HOST}/{INDEX_NAME}/_search"
    
    # Match query with operator "or" and 60% minimum_should_match
    match_query = {
        "asr_text": {
            "query": query,
            "operator": "or",
            "minimum_should_match": "60%"
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
        "_source": ["video_id", "keyframe_id", "frame_idx", "image_path", "asr_text"]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            logger.error("Elasticsearch ASR search failed with status %d: %s", response.status_code, response.text)
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
                "asr_text": source.get("asr_text", "")
            })
        return results
    except Exception as exc:
        logger.exception("Error searching ASR in Elasticsearch: %s", exc)
        return []


def get_asr_for_keyframe(video_id: str, frame_idx: int, keyframe_id: Optional[str] = None) -> Optional[str]:
    """
    Get the ASR segment (transcript) for a specific keyframe.
    Tries keyframe_id query first, falls back to video_id and frame_idx.
    """
    url = f"{ES_HOST}/{INDEX_NAME}/_search"
    
    # 1. Query by keyframe_id first
    if keyframe_id:
        payload = {
            "query": {
                "term": {
                    "keyframe_id.keyword": keyframe_id
                }
            },
            "size": 1,
            "_source": ["asr_text"]
        }
        try:
            response = requests.post(url, json=payload, timeout=3)
            if response.status_code == 200:
                hits = response.json().get("hits", {}).get("hits", [])
                if hits:
                    return hits[0].get("_source", {}).get("asr_text", "")
        except Exception as e:
            logger.warning(f"Error querying ASR by keyframe_id {keyframe_id}: {e}")
            
    # 2. Fallback to video_id + frame_idx
    payload = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"video_id.keyword": video_id}},
                    {"term": {"frame_idx": int(frame_idx)}}
                ]
            }
        },
        "size": 1,
        "_source": ["asr_text"]
    }
    try:
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 200:
            hits = response.json().get("hits", {}).get("hits", [])
            if hits:
                return hits[0].get("_source", {}).get("asr_text", "")
    except Exception as e:
        logger.warning(f"Error querying ASR by video_id {video_id} and frame_idx {frame_idx}: {e}")
        
    return None

