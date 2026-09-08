import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("asr_metadata")

_PRIMARY_ASR_DIR = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr")
_FALLBACK_ASR_DIR = Path("/dev/shm/asr")
ASR_METADATA_ROOT = _PRIMARY_ASR_DIR if _PRIMARY_ASR_DIR.exists() else _FALLBACK_ASR_DIR

# In-memory RAM caches for 100% preloaded ASR metadata
_ASR_KEYFRAME_CACHE: Dict[str, Dict[str, Any]] = {}
_ASR_VIDEO_RECORDS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_ASR_PRELOADED: bool = False
_ASR_PRELOAD_LOCK = threading.Lock()


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    return records


def _normalize_asr_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("asr_metadata") or {}
    asr_6s = str(metadata.get("asr_text_6s") or "").strip()
    raw_segments = metadata.get("raw_segments") or []
    full_text = " ".join(
        str(seg.get("text") or "").strip()
        for seg in raw_segments
        if isinstance(seg, dict) and (seg.get("text") or "").strip()
    ).strip()

    return {
        "video_id": record.get("video_id"),
        "keyframe_id": record.get("keyframe_id"),
        "frame_idx": record.get("frame_idx"),
        "asr_text": asr_6s,
        "asr_6s": asr_6s,
        "full_text": full_text,
        "window_start": metadata.get("window_start"),
        "window_end": metadata.get("window_end"),
        "has_speech": bool(metadata.get("has_speech", bool(asr_6s))),
    }


from concurrent.futures import ThreadPoolExecutor

def _process_single_asr_file(p: Path) -> Tuple[str, List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]:
    video_id = p.name.replace("_asr.jsonl", "")
    raw_records = _read_jsonl_records(p)
    if not raw_records:
        return video_id, [], []

    norm_records = []
    kf_entries = []
    for raw in raw_records:
        norm = _normalize_asr_record(raw)
        norm_records.append(norm)
        kf_id = norm.get("keyframe_id")
        if kf_id:
            kf_entries.append((kf_id, norm))

    return video_id, norm_records, kf_entries

def preload_asr_metadata() -> int:
    """
    Preload 100% of ASR metadata JSONL files into RAM cache at startup.
    Returns the total number of preloaded keyframes.
    """
    global _ASR_PRELOADED
    with _ASR_PRELOAD_LOCK:
        if _ASR_PRELOADED:
            return len(_ASR_KEYFRAME_CACHE)

        t0 = time.perf_counter()
        if not ASR_METADATA_ROOT.exists():
            logger.warning(f"ASR metadata directory {ASR_METADATA_ROOT} does not exist.")
            _ASR_PRELOADED = True
            return 0

        jsonl_files = list(ASR_METADATA_ROOT.glob("*_asr.jsonl"))
        kf_count = 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = executor.map(_process_single_asr_file, jsonl_files)
            for video_id, norm_records, kf_entries in results:
                if norm_records:
                    _ASR_VIDEO_RECORDS_CACHE[video_id] = norm_records
                for kf_id, norm in kf_entries:
                    _ASR_KEYFRAME_CACHE[kf_id] = norm
                    kf_count += 1

        _ASR_PRELOADED = True
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Preloaded {len(jsonl_files)} ASR files ({kf_count} keyframes) into RAM in {elapsed:.2f}s"
        )
        return kf_count


def get_asr_for_keyframe_metadata(
    video_id: str,
    frame_idx: Optional[int] = None,
    keyframe_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not video_id:
        return None

    # 1. Instant O(1) RAM lookup if keyframe_id is present in _ASR_KEYFRAME_CACHE
    if keyframe_id and keyframe_id in _ASR_KEYFRAME_CACHE:
        return _ASR_KEYFRAME_CACHE[keyframe_id]

    # 2. Check video records in RAM cache
    records = _ASR_VIDEO_RECORDS_CACHE.get(video_id)

    # 3. Fallback: Lazy load if cache miss
    if records is None:
        metadata_path = ASR_METADATA_ROOT / f"{video_id}_asr.jsonl"
        if not metadata_path.exists():
            return None
        raw_records = _read_jsonl_records(metadata_path)
        if not raw_records:
            return None
        records = [_normalize_asr_record(r) for r in raw_records]
        _ASR_VIDEO_RECORDS_CACHE[video_id] = records
        for r in records:
            if r.get("keyframe_id"):
                _ASR_KEYFRAME_CACHE[r["keyframe_id"]] = r

    if not records:
        return None

    if keyframe_id:
        for record in records:
            if record.get("keyframe_id") == keyframe_id:
                return record

    target_frame = None
    if frame_idx is not None:
        try:
            target_frame = int(frame_idx)
        except (TypeError, ValueError):
            target_frame = None

    if target_frame is not None:
        exact_match = None
        nearest_match = None
        nearest_diff = None

        for record in records:
            try:
                record_frame = int(record.get("frame_idx"))
            except (TypeError, ValueError):
                continue

            if record_frame == target_frame:
                exact_match = record
                break

            diff = abs(record_frame - target_frame)
            if nearest_diff is None or diff < nearest_diff:
                nearest_diff = diff
                nearest_match = record

        if exact_match is not None:
            return exact_match
        if nearest_match is not None:
            return nearest_match

    if keyframe_id:
        for record in records:
            if record.get("keyframe_id", "").endswith(keyframe_id.split("_")[-1]):
                return record

    return records[0]
