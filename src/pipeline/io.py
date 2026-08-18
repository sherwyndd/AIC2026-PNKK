import json
import os
from pathlib import Path

import pandas as pd


def save_metadata(data, output_path, out_format):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    if out_format == "csv":
        df.to_csv(output_path, index=False)
    elif out_format == "json":
        df.to_json(output_path, orient="records", indent=4, force_ascii=False)
    elif out_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in df.to_dict(orient="records"):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {out_format}")


def _read_metadata(path):
    if not os.path.exists(path):
        return None
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata file extension: {suffix}")


def load_metadata_table(path_csv, path_json=None):
    table = _read_metadata(path_csv)
    if table is not None:
        return table
    if path_json:
        return _read_metadata(path_json)
    return None


def load_video_list(list_path):
    if list_path is None:
        return None
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"Video list file not found: {list_path}")
    with open(list_path, "r", encoding="utf-8") as handle:
        raw_lines = [line.strip() for line in handle.readlines()]
    videos = []
    for line in raw_lines:
        if not line or line.startswith("#"):
            continue
        videos.append(line)
    return videos


def _normalize_video_list_entry(entry):
    if not entry:
        return None
    entry = entry.strip()
    if not entry:
        return None
    if entry.startswith("#"):
        return None
    return os.path.basename(entry)


def list_videos(input_dir, mode, video_list=None):
    videos = [
        f
        for f in sorted(os.listdir(input_dir))
        if f.lower().endswith((".mp4", ".mkv", ".avi"))
    ]
    if video_list:
        existing = set(videos)
        normalized = []
        for item in video_list:
            name = _normalize_video_list_entry(item)
            if name and name in existing:
                normalized.append(name)
        videos = normalized
    if mode == "demo" and videos:
        videos = videos[:1]
    return videos


def ensure_output_dirs(output_dir, layout):
    metadata_dir = layout["metadata_dir"]
    keyframes_dir = layout["keyframes_dir"]
    paths = [
        output_dir,
        os.path.join(output_dir, metadata_dir, "videos"),
        os.path.join(output_dir, metadata_dir, "shots"),
        os.path.join(output_dir, metadata_dir, "keyframes"),
        os.path.join(output_dir, metadata_dir, "asr"),
        os.path.join(output_dir, metadata_dir, "captions"),
        os.path.join(output_dir, metadata_dir, "objects"),
        os.path.join(output_dir, keyframes_dir),
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)


def metadata_paths(output_dir, video_id, out_format, layout):
    metadata_dir = layout["metadata_dir"]
    return {
        "videos": os.path.join(output_dir, metadata_dir, "videos", f"videos_metadata.{out_format}"),
        "shots": os.path.join(output_dir, metadata_dir, "shots", f"{video_id}_shots.{out_format}"),
        "keyframes": os.path.join(
            output_dir, metadata_dir, "keyframes", f"{video_id}_keyframes.{out_format}"
        ),
    }


def asr_metadata_path(output_dir, video_id, out_format, layout):
    metadata_dir = layout["metadata_dir"]
    return os.path.join(output_dir, metadata_dir, "asr", f"{video_id}_asr.{out_format}")


def object_metadata_path(output_dir, video_id, out_format, layout):
    metadata_dir = layout["metadata_dir"]
    return os.path.join(output_dir, metadata_dir, "objects", f"{video_id}_objects.{out_format}")


def caption_metadata_path(output_dir, video_id, out_format, layout):
    metadata_dir = layout["metadata_dir"]
    return os.path.join(output_dir, metadata_dir, "captions", f"{video_id}_captions.{out_format}")


def keyframes_video_dir(output_dir, video_id, layout):
    return os.path.join(output_dir, layout["keyframes_dir"], video_id)


def keyframe_image_relpath(video_id, keyframe_id, layout):
    return f"{layout['keyframes_dir']}/{video_id}/{keyframe_id}.jpg"
