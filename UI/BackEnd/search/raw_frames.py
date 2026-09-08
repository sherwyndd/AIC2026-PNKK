"""
Raw frame extraction module for competition precision.
Extracts every exact raw frame (25-75 frames) in a 1-3 second window from the original MP4 video.
"""

import os
import cv2
import base64
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

VIDEO_SEARCH_DIRS = [
    "/AIClub_NAS/core_baotg/nhan/dataset/keyframes",
    "/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos",
    "/AIClub_NAS/core_baotg/nhan/dataset",
    "/workspace/dataset/HCMAI25/full",
    "/workspace/dataset/HCMAI25/batch1",
    "/workspace/dataset/HCMAI25/streaming",
    "/workspace/dataset/Viettel_AI_Track_1",
    "/workspace/dataset/keyframes",
]

_VIDEO_PATH_CACHE: Dict[str, str] = {}


def find_video_path(video_id: str) -> Optional[str]:
    """Find the absolute path of an MP4 video file by video_id."""
    if video_id in _VIDEO_PATH_CACHE:
        path = _VIDEO_PATH_CACHE[video_id]
        if os.path.isfile(path):
            return path

    for base_dir in VIDEO_SEARCH_DIRS:
        candidate = os.path.join(base_dir, f"{video_id}.mp4")
        if os.path.isfile(candidate):
            _VIDEO_PATH_CACHE[video_id] = candidate
            return candidate

    return None


def extract_raw_frames(
    video_id: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    center_sec: Optional[float] = None,
    duration_sec: float = 2.0,
    target_height: int = 360,
    jpeg_quality: int = 80,
    max_duration: float = 5.0
) -> Dict[str, Any]:
    """
    Extract all raw individual frames within a short time range (1-3 seconds).

    Args:
        video_id: Video identifier (e.g. 'L21_V001')
        start_sec: Start time in seconds (optional)
        end_sec: End time in seconds (optional)
        center_sec: Center time in seconds (optional, used if start_sec is None)
        duration_sec: Duration in seconds (default 2.0s, clamped between 0.5s and max_duration)
        target_height: Height of output thumbnails (default 360px for sharpness & fast transfer)
        jpeg_quality: JPEG compression quality (default 80)
        max_duration: Maximum allowed duration to prevent excessive memory/payload size

    Returns:
        Dict containing video info, fps, frame count, and list of frame dicts with base64 data URLs.
    """
    video_path = find_video_path(video_id)
    if not video_path:
        raise FileNotFoundError(f"Video file not found for video_id '{video_id}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration = total_frames / fps if fps > 0 else 0.0

        # Calculate time range
        if start_sec is not None and end_sec is not None:
            s_sec = max(0.0, float(start_sec))
            e_sec = max(s_sec + 0.1, float(end_sec))
            # Clamp duration to max_duration
            if (e_sec - s_sec) > max_duration:
                e_sec = s_sec + max_duration
        elif center_sec is not None:
            dur = max(0.5, min(float(duration_sec), max_duration))
            half = dur / 2.0
            s_sec = max(0.0, float(center_sec) - half)
            e_sec = s_sec + dur
        elif start_sec is not None:
            dur = max(0.5, min(float(duration_sec), max_duration))
            s_sec = max(0.0, float(start_sec))
            e_sec = s_sec + dur
        else:
            s_sec = 0.0
            e_sec = min(float(duration_sec), max_duration)

        if total_duration > 0 and e_sec > total_duration:
            e_sec = total_duration

        start_frame = max(0, int(round(s_sec * fps)))
        end_frame = min(total_frames - 1, int(round(e_sec * fps))) if total_frames > 0 else int(round(e_sec * fps))

        if start_frame > end_frame:
            start_frame = end_frame

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        curr_frame = start_frame

        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if target_height > 0 and h > target_height:
                scale = target_height / float(h)
                new_w = int(round(w * scale))
                resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)
            else:
                resized = frame

            success, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if success:
                b64_str = base64.b64encode(buf).decode("ascii")
                data_url = f"data:image/jpeg;base64,{b64_str}"
            else:
                data_url = ""

            pts_time = round(curr_frame / fps, 3)
            timestamp_ms = int(round((curr_frame / fps) * 1000))

            frames.append({
                "frame_idx": curr_frame,
                "pts_time": pts_time,
                "timestamp": timestamp_ms,
                "image_url": data_url,
                "keyframe_id": f"{video_id}_f{curr_frame}",
                "width": resized.shape[1],
                "height": resized.shape[0],
            })
            curr_frame += 1

        return {
            "status": "success",
            "video_id": video_id,
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "total_duration": round(total_duration, 3),
            "start_time": round(s_sec, 3),
            "end_time": round(e_sec, 3),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "count": len(frames),
            "frames": frames,
        }
    finally:
        cap.release()
