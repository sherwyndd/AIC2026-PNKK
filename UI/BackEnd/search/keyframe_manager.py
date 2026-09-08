"""
Keyframe metadata and timestamp management for ultra-fast competition retrieval.
Loads and caches keyframe metadata from dataset CSVs and keyframe directories.

CHANGELOG (bugfix):
    - [FIX] get_keyframe_timestamp() trước đây, khi không tìm thấy
      keyframe_id/frame_idx khớp trong danh sách, sẽ ÂM THẦM fallback về
      timestamp của item ĐẦU TIÊN trong video (kfs[0]["timestamp"]) thay vì
      báo lỗi hoặc trả 0. Giá trị này được dùng trực tiếp để submit lên
      DRES — lookup sai mà vẫn trả về một số "hợp lý" rất dễ khiến nộp
      sai timestamp mà không hề hay biết. Đã bỏ fallback này, trả về 0
      kèm log warning khi không khớp được.
    - [FIX] Thêm lock theo từng video_id trong _load_video_metadata() để
      tránh nhiều thread cùng lúc cache-miss rồi cùng đọc/ghi đè CSV song
      song (lãng phí I/O trên NAS) khi nhiều lookup cùng video chạy đồng
      thời qua asyncio.gather ở main.py.
    - [CLEANUP] Bỏ import functools.lru_cache không dùng tới.
"""

import csv
import logging
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("keyframe_manager")

METADATA_KEYFRAMES_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes"
KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
VIDEOS_METADATA_CSV = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv"

_VIDEOS_METADATA_MAP: Dict[str, Dict] = {}
_VIDEOS_FPS_MAP: Dict[str, float] = {}


def load_videos_metadata() -> Dict[str, Dict]:
    """
    Load videos_metadata.csv into memory cache:
    video_id -> {"video_id": str, "fps": float, "total_frames": int, "duration": float, ...}
    """
    global _VIDEOS_METADATA_MAP, _VIDEOS_FPS_MAP
    if _VIDEOS_METADATA_MAP:
        return _VIDEOS_METADATA_MAP

    if os.path.isfile(VIDEOS_METADATA_CSV):
        try:
            with open(VIDEOS_METADATA_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    vid = row.get("video_id", "").strip()
                    if not vid:
                        continue
                    try:
                        fps = float(row.get("fps", 25.0))
                    except (ValueError, TypeError):
                        fps = 25.0
                    try:
                        total_frames = int(row.get("total_frames", 0))
                    except (ValueError, TypeError):
                        total_frames = 0
                    try:
                        duration = float(row.get("duration", 0.0))
                    except (ValueError, TypeError):
                        duration = 0.0

                    meta = {
                        "video_id": vid,
                        "file_name": row.get("file_name", f"{vid}.mp4"),
                        "fps": fps,
                        "total_frames": total_frames,
                        "duration": duration,
                        "width": int(row.get("width", 1280)) if str(row.get("width", "")).isdigit() else 1280,
                        "height": int(row.get("height", 720)) if str(row.get("height", "")).isdigit() else 720,
                        "bitrate_kbps": float(row.get("bitrate_kbps", 0.0)) if row.get("bitrate_kbps") else 0.0,
                    }
                    _VIDEOS_METADATA_MAP[vid] = meta
                    _VIDEOS_FPS_MAP[vid] = fps
                    _VIDEOS_FPS_MAP[f"{vid}.mp4"] = fps
                    _VIDEOS_FPS_MAP[vid.lower()] = fps
            logger.info(f"Loaded {len(_VIDEOS_METADATA_MAP)} video metadata records from {VIDEOS_METADATA_CSV}")
        except Exception as e:
            logger.error(f"Error loading {VIDEOS_METADATA_CSV}: {e}")

    return _VIDEOS_METADATA_MAP


def get_video_fps(video_id: str) -> float:
    """Return video FPS from videos_metadata.csv (fallback to 25.0)."""
    if not _VIDEOS_FPS_MAP:
        load_videos_metadata()
    clean_vid = str(video_id).replace(".mp4", "").strip()
    return _VIDEOS_FPS_MAP.get(clean_vid, _VIDEOS_FPS_MAP.get(video_id, 25.0))


def get_video_info(video_id: str) -> Optional[Dict]:
    """Return video metadata dict from videos_metadata.csv."""
    if not _VIDEOS_METADATA_MAP:
        load_videos_metadata()
    clean_vid = str(video_id).replace(".mp4", "").strip()
    return _VIDEOS_METADATA_MAP.get(clean_vid)


def get_all_videos_fps() -> Dict[str, float]:
    """Return dictionary of all {video_id: fps}."""
    if not _VIDEOS_FPS_MAP:
        load_videos_metadata()
    return {k: v for k, v in _VIDEOS_FPS_MAP.items() if not k.endswith(".mp4") and not k.islower()}

# Prefixes to strip from legacy absolute image_path values
# (kept in sync with main.py's normalize_image_path)
_PATH_PREFIXES_TO_STRIP = (
    "/AIClub_NAS/core_baotg/nhan/dataset/keyframes/",
    "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes/",
    "/AIClub_NAS/core_baotg/nhan/dataset/",
    "/AIClub_NAS/Datasets/HCMAI25/full/",
    "/AIClub_NAS/Datasets/Viettel_AI_Track_1/",
    "/mlcv1/Datasets/HCMAI25/full/",
    "/mlcv1/Datasets/HCMAI25/batch1/",
    "/mlcv1/Datasets/HCMAI25/streaming/",
    "/workspace/dataset/keyframes/",
    "/workspace/dataset/metadata/keyframes/",
    "/workspace/dataset/HCMAI25/full/",
    "/workspace/dataset/HCMAI25/batch1/",
    "/workspace/dataset/HCMAI25/streaming/",
    "/workspace/dataset/Viettel_AI_Track_1/",
    "/workspace/dataset/",
    "keyframes/",
)


def _normalize_rel_image_path(raw: str, video_id: str, kf_id: str) -> str:
    """Strip any legacy absolute dataset prefixes from a raw image_path.

    Falls back to {video_id}/{kf_id}.jpg if the result is empty.
    """
    if not raw:
        return f"{video_id}/{kf_id}.jpg"
    p = raw.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _PATH_PREFIXES_TO_STRIP:
            if p.startswith(prefix):
                p = p[len(prefix):]
                changed = True
                break
            elif p.startswith("/" + prefix):
                p = p[len(prefix) + 1:]
                changed = True
                break
    p = p.lstrip("/")
    return p or f"{video_id}/{kf_id}.jpg"

# In-memory caches for fast retrieval
_VIDEO_KEYFRAMES_CACHE: Dict[str, List[Dict]] = {}
_TIMESTAMP_LOOKUP: Dict[str, int] = {}  # keyframe_id -> timestamp (ms)

# Per-video lock to avoid duplicate concurrent CSV reads (race condition
# khi nhiều thread cùng cache-miss 1 video cùng lúc).
_LOAD_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _get_video_lock(video_id: str) -> threading.Lock:
    """Trả về (hoặc tạo mới) lock riêng cho video_id này."""
    with _LOCKS_GUARD:
        lock = _LOAD_LOCKS.get(video_id)
        if lock is None:
            lock = threading.Lock()
            _LOAD_LOCKS[video_id] = lock
        return lock


def _load_video_metadata(video_id: str) -> List[Dict]:
    """
    Load all keyframes for a video from its CSV or directory.
    Thread-safe: dùng lock riêng theo video_id để tránh nhiều thread cùng
    đọc CSV song song khi cache-miss đồng thời.

    Returns list of dicts: [
        {
            "keyframe_id": "L21_V001_kf_0001",
            "shot_id": "L21_V001_shot_001",
            "video_id": "L21_V001",
            "frame_idx": 0,
            "pts_time": 0.0,
            "timestamp": 0,
            "image_path": "L21_V001/L21_V001_kf_0001.jpg",
            "image_url": "/images/L21_V001/L21_V001_kf_0001.jpg"
        },
        ...
    ]
    """
    if video_id in _VIDEO_KEYFRAMES_CACHE:
        return _VIDEO_KEYFRAMES_CACHE[video_id]

    with _get_video_lock(video_id):
        # Double-checked locking: sau khi có lock, kiểm tra lại cache —
        # có thể thread khác đã load xong trong lúc mình chờ lock.
        if video_id in _VIDEO_KEYFRAMES_CACHE:
            return _VIDEO_KEYFRAMES_CACHE[video_id]

        csv_path = os.path.join(METADATA_KEYFRAMES_DIR, f"{video_id}_keyframes.csv")
        keyframes: List[Dict] = []

        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        kf_id = row.get("keyframe_id", "").strip()
                        shot_id = row.get("shot_id", "").strip()
                        v_id = row.get("video_id", video_id).strip()

                        try:
                            frame_idx = int(row.get("frame_idx", 0))
                        except (ValueError, TypeError):
                            frame_idx = 0

                        try:
                            pts_time = float(row.get("pts_time", 0.0))
                        except (ValueError, TypeError):
                            pts_time = 0.0

                        timestamp_ms = int(round(pts_time * 1000))

                        raw_img_path = row.get("image_path", "").strip()
                        rel_img_path = _normalize_rel_image_path(raw_img_path, video_id, kf_id)

                        item = {
                            "keyframe_id": kf_id,
                            "shot_id": shot_id,
                            "video_id": v_id,
                            "frame_idx": frame_idx,
                            "pts_time": pts_time,
                            "timestamp": timestamp_ms,
                            "image_path": rel_img_path,
                            "image_url": f"/images/{rel_img_path}",
                        }
                        keyframes.append(item)
                        _TIMESTAMP_LOOKUP[kf_id] = timestamp_ms
                        _TIMESTAMP_LOOKUP[f"{v_id}:{frame_idx}"] = timestamp_ms
            except Exception as e:
                logger.warning(f"Error reading CSV for video {video_id}: {e}")

        # Fallback to image directory if CSV not available or empty
        if not keyframes:
            vid_dir = os.path.join(KEYFRAME_ROOT, video_id)
            if os.path.isdir(vid_dir):
                try:
                    files = sorted(f for f in os.listdir(vid_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                    real_fps = get_video_fps(video_id)
                    for idx, fname in enumerate(files):
                        kf_id = os.path.splitext(fname)[0]
                        # Estimate pts_time using real FPS from videos_metadata.csv
                        pts_time = idx * (1.0 / real_fps)
                        timestamp_ms = int(round(pts_time * 1000))
                        rel_img_path = f"{video_id}/{fname}"
                        item = {
                            "keyframe_id": kf_id,
                            "shot_id": "",
                            "video_id": video_id,
                            "frame_idx": idx,
                            "pts_time": pts_time,
                            "timestamp": timestamp_ms,
                            "image_path": rel_img_path,
                            "image_url": f"/images/{rel_img_path}",
                        }
                        keyframes.append(item)
                        _TIMESTAMP_LOOKUP[kf_id] = timestamp_ms
                        _TIMESTAMP_LOOKUP[f"{video_id}:{idx}"] = timestamp_ms
                except Exception as e:
                    logger.warning(f"Error listing directory for video {video_id}: {e}")

        if not keyframes:
            logger.warning(
                "Không tìm thấy metadata nào cho video '%s' (không có CSV lẫn thư mục ảnh).",
                video_id,
            )

        _VIDEO_KEYFRAMES_CACHE[video_id] = keyframes
        return keyframes


def get_video_keyframes(video_id: str) -> List[Dict]:
    """Get all keyframes for a video (cached)."""
    return _load_video_metadata(video_id)


def get_keyframe_timestamp(
    video_id: str,
    keyframe_id: Optional[str] = None,
    frame_idx: Optional[int] = None
) -> int:
    """
    Ultra-fast timestamp resolution in milliseconds.
    Checks lookup cache first, then loads video metadata if needed.

    [FIX] Không còn fallback về timestamp của keyframe đầu tiên trong video
    khi không tìm thấy khớp — giá trị này dùng trực tiếp để submit DRES,
    trả về một con số "trông hợp lý" nhưng sai sẽ rất khó phát hiện.
    Nếu không tìm được, trả về 0 và ghi log warning để dễ debug.
    """
    if keyframe_id and keyframe_id in _TIMESTAMP_LOOKUP:
        return _TIMESTAMP_LOOKUP[keyframe_id]

    if video_id and frame_idx is not None:
        key = f"{video_id}:{frame_idx}"
        if key in _TIMESTAMP_LOOKUP:
            return _TIMESTAMP_LOOKUP[key]

    # Ensure video is loaded
    if video_id:
        kfs = _load_video_metadata(video_id)

        if keyframe_id:
            for item in kfs:
                if item["keyframe_id"] == keyframe_id:
                    return item["timestamp"]

        if frame_idx is not None:
            for item in kfs:
                if item["frame_idx"] == frame_idx:
                    return item["timestamp"]

        # [FIX] KHÔNG fallback về kfs[0] nữa — trả 0 + log rõ ràng thay vì
        # âm thầm trả một timestamp sai nhưng trông "hợp lý".
        logger.warning(
            "Không tìm thấy timestamp khớp cho video=%s keyframe_id=%r frame_idx=%r "
            "(video có %d keyframe trong cache) — trả về 0.",
            video_id, keyframe_id, frame_idx, len(kfs),
        )
        return 0

    return 0


def resolve_keyframe_metadata(
    video_id: str,
    keyframe_id: Optional[str] = None,
    raw_frame_idx: Optional[int] = None
) -> Tuple[int, int, float, str]:
    """
    Given video_id, keyframe_id (e.g. L25_V084_kf_0176 or 176) and optional raw_frame_idx:
    Looks up the true frame_idx in the dataset CSV.
    Returns (real_frame_idx, timestamp_ms, pts_time_s, image_path).
    """
    clean_vid = str(video_id).replace(".mp4", "").strip() if video_id else ""
    if clean_vid:
        kfs = _load_video_metadata(clean_vid)
        if keyframe_id:
            # 1. Exact match by keyframe_id
            for item in kfs:
                if item["keyframe_id"] == keyframe_id:
                    return int(item["frame_idx"]), int(item["timestamp"]), float(item["pts_time"]), str(item["image_path"])

            # 2. Match by numeric suffix in keyframe_id (e.g. 176 or kf_0176)
            m = re.search(r'(\d+)$', str(keyframe_id))
            if m:
                num_suffix = int(m.group(1))
                for item in kfs:
                    im = re.search(r'(\d+)$', item["keyframe_id"])
                    if im and int(im.group(1)) == num_suffix:
                        return int(item["frame_idx"]), int(item["timestamp"]), float(item["pts_time"]), str(item["image_path"])

        if raw_frame_idx is not None:
            # 3. Exact match by frame_idx
            for item in kfs:
                if item["frame_idx"] == raw_frame_idx:
                    return int(item["frame_idx"]), int(item["timestamp"]), float(item["pts_time"]), str(item["image_path"])

            # 4. Check if raw_frame_idx matches keyframe sequence number (e.g. 176)
            for item in kfs:
                im = re.search(r'(\d+)$', item["keyframe_id"])
                if im and int(im.group(1)) == raw_frame_idx:
                    return int(item["frame_idx"]), int(item["timestamp"]), float(item["pts_time"]), str(item["image_path"])

            # 5. Fallback: closest keyframe by frame_idx
            if kfs:
                closest = min(kfs, key=lambda k: abs(int(k.get("frame_idx", 0)) - raw_frame_idx))
                return int(closest["frame_idx"]), int(closest["timestamp"]), float(closest["pts_time"]), str(closest["image_path"])

    ts_ms = get_keyframe_timestamp(clean_vid, keyframe_id, raw_frame_idx)
    pts_s = round(ts_ms / 1000.0, 3)
    fidx = raw_frame_idx if raw_frame_idx is not None else 0
    return fidx, ts_ms, pts_s, f"{clean_vid}/{keyframe_id}.jpg" if keyframe_id else ""


# ── Video path finder ──────────────────────────────────────────────────────────
VIDEO_SEARCH_DIRS = [
    "/AIClub_NAS/Datasets/HCMAI25/full",
    "/AIClub_NAS/Datasets/HCMAI25/batch1",
    "/AIClub_NAS/Datasets/HCMAI25",
    "/AIClub_NAS/core_baotg/nhan/dataset/videos",
    "/AIClub_NAS/core_baotg/nhan/dataset",
    "/AIClub_NAS/Datasets/Viettel_AI_Track_1",
    "/mlcv1/Datasets/HCMAI25/full",
    "/mlcv1/Datasets/HCMAI25/batch1",
    "/mlcv1/Datasets/HCMAI25/streaming",
]

_VIDEO_PATH_CACHE: Dict[str, str] = {}


def find_video_path(video_id: str) -> Optional[str]:
    """Find the absolute path of an MP4 video file by video_id."""
    clean_id = video_id.replace(".mp4", "").strip()
    if clean_id in _VIDEO_PATH_CACHE:
        path = _VIDEO_PATH_CACHE[clean_id]
        if os.path.isfile(path):
            return path

    for base_dir in VIDEO_SEARCH_DIRS:
        candidate = os.path.join(base_dir, f"{clean_id}.mp4")
        if os.path.isfile(candidate):
            _VIDEO_PATH_CACHE[clean_id] = candidate
            return candidate

    return None


from typing import Tuple

def parse_video_frame_id(input_str: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Parse input strings like:
      - 'L21_V001_12345' -> ('L21_V001', 12345, None)
      - 'L21_V001 12345' -> ('L21_V001', 12345, None)
      - 'L21_V001/12345' -> ('L21_V001', 12345, None)
      - 'L21_V001:12345' -> ('L21_V001', 12345, None)
      - 'L21_V001_kf_0045' -> ('L21_V001', None, 'L21_V001_kf_0045')
      - 'L21_V001' -> ('L21_V001', None, None)
    """
    if not input_str:
        return "", None, None
    s = str(input_str).strip()

    if "_kf_" in s:
        parts = s.split("_kf_")
        vid = parts[0].strip()
        return vid, None, s

    for sep in [":", "/", ",", "#", " ", "\t"]:
        s = s.replace(sep, "_")

    while "__" in s:
        s = s.replace("__", "_")

    if "_" in s:
        left, right = s.rsplit("_", 1)
        if right.isdigit():
            return left.strip(), int(right), None
        return s, None, None

    return s, None, None


def get_nearest_keyframe(
    video_id: str,
    target_frame_idx: Optional[int] = None,
    target_keyframe_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Find the keyframe in video_id that is closest to target_frame_idx (or matching target_keyframe_id).
    Returns the keyframe dict with keys:
      keyframe_id, video_id, frame_idx, pts_time, timestamp, image_path, image_url.
    """
    clean_vid = str(video_id).replace(".mp4", "").strip()
    if not clean_vid:
        return None
    kfs = get_video_keyframes(clean_vid)
    if not kfs:
        return None

    if target_keyframe_id:
        target_kfid_clean = target_keyframe_id.strip()
        for k in kfs:
            if k.get("keyframe_id") == target_kfid_clean:
                return k
        for k in kfs:
            if str(k.get("keyframe_id", "")).endswith(target_kfid_clean):
                return k

    if target_frame_idx is not None:
        for k in kfs:
            if k.get("frame_idx") == target_frame_idx:
                return k
        return min(kfs, key=lambda k: abs(int(k.get("frame_idx", 0)) - target_frame_idx))

    return kfs[0]