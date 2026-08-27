#!/usr/bin/env python3
"""
Sinh lại videos_metadata.json (fps, duration, total_frames, width, height, ...)
cho toàn bộ video — KHÔNG cắt shot, KHÔNG cần GPU.
"""

import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_pipeline_defaults, load_runtime_config, resolve_path
from src.pipeline.transnet import get_video_metadata

# ── Config ──────────────────────────────────────────────────────────────────
config = load_runtime_config()
defaults = get_pipeline_defaults(config)

INPUT_DIR = defaults["input_dir"]
OUTPUT_DIR = resolve_path(config.get("paths", {}).get("output_full", "data"))
OUTPUT_FILE = OUTPUT_DIR / "metadata" / "videos" / "videos_metadata.json"

# ── List videos ──────────────────────────────────────────────────────────────
video_exts = (".mp4", ".mkv", ".avi", ".mov")
videos = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(video_exts)
])

print(f"Input dir : {INPUT_DIR}")
print(f"Output    : {OUTPUT_FILE}")
print(f"Videos    : {len(videos)} found")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Load existing metadata (nếu có) để merge ────────────────────────────────
existing = {}
if OUTPUT_FILE.exists():
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        try:
            for rec in json.load(f):
                existing[rec["video_id"]] = rec
    print(f"Loaded {len(existing)} existing records")

# ── Collect metadata ─────────────────────────────────────────────────────────
all_metadata = {}
errors = []

for video_file in tqdm(videos, desc="Collecting metadata"):
    video_id = os.path.splitext(video_file)[0]
    video_path = os.path.join(INPUT_DIR, video_file)

    meta = get_video_metadata(video_path, video_id)
    if meta:
        all_metadata[video_id] = meta
    else:
        errors.append(video_file)
        print(f"  [WARN] Cannot open: {video_file}")

# Merge: ưu tiên kết quả mới, giữ lại record cũ nếu video không còn trong INPUT_DIR
merged = {**existing, **all_metadata}
records = list(merged.values())

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"\nDone. Saved {len(records)} records → {OUTPUT_FILE}")
if errors:
    print(f"Errors ({len(errors)}): {errors}")
