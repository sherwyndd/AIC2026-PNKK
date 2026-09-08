#!/usr/bin/env python3
"""Extract video technical metadata into videos_metadata.csv.

Schema matches:
  /AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv

  video_id, file_name, fps, total_frames, duration, width, height,
  bitrate_kbps, backend, format, fourcc

Example:
  python scripts/extract_videos_metadata.py \\
    --input_dir /mlcv2025/Datasets/HCMAI25/batch2/video \\
    --output /AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv \\
    --existing /AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
CSV_COLUMNS = [
    "video_id",
    "file_name",
    "fps",
    "total_frames",
    "duration",
    "width",
    "height",
    "bitrate_kbps",
    "backend",
    "format",
    "fourcc",
]


def _fourcc_to_str(code: int) -> str:
    if not code:
        return ""
    chars = "".join(chr((int(code) >> (8 * i)) & 0xFF) for i in range(4)).strip()
    return chars.replace("\x00", "")


def probe_video(video_path: str) -> dict | None:
    import cv2

    file_name = os.path.basename(video_path)
    video_id = os.path.splitext(file_name)[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"video_id": video_id, "file_name": file_name, "error": "cannot_open"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = round(total_frames / fps, 6) if fps > 0 else 0.0
    bitrate = cap.get(cv2.CAP_PROP_BITRATE)
    backend = cap.getBackendName()
    fmt = int(cap.get(cv2.CAP_PROP_FORMAT) or 0)
    fourcc = _fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC) or 0))
    cap.release()

    bitrate_kbps = ""
    if bitrate and bitrate > 0:
        bitrate_kbps = round(float(bitrate), 1)

    return {
        "video_id": video_id,
        "file_name": file_name,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
        "bitrate_kbps": bitrate_kbps,
        "backend": backend,
        "format": fmt if fmt else "",
        "fourcc": fourcc,
    }


def list_video_files(input_dir: str, recursive: bool) -> list[str]:
    paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for name in files:
                if Path(name).suffix.lower() in VIDEO_EXTS:
                    paths.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            full = os.path.join(input_dir, name)
            if os.path.isfile(full) and Path(name).suffix.lower() in VIDEO_EXTS:
                paths.append(full)
    return sorted(paths)


def load_existing_ids(csv_path: str) -> set[str]:
    if not csv_path or not os.path.isfile(csv_path):
        return set()
    ids = set()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vid = (row.get("video_id") or "").strip()
            if vid:
                ids.add(vid)
    return ids


def load_video_list(list_path: str | None) -> set[str] | None:
    if not list_path:
        return None
    names = set()
    with open(list_path, encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            names.add(os.path.splitext(os.path.basename(name))[0])
            names.add(os.path.basename(name))
    return names


def write_csv(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r.get("video_id") or "")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})


def merge_existing(existing_path: str, new_rows: list[dict]) -> list[dict]:
    merged = {}
    if existing_path and os.path.isfile(existing_path):
        with open(existing_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                vid = row.get("video_id")
                if vid:
                    merged[vid] = row
    for row in new_rows:
        vid = row.get("video_id")
        if vid:
            merged[vid] = row
    return list(merged.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe videos and write videos_metadata.csv (same schema as nhan dataset)."
    )
    parser.add_argument(
        "--input_dir",
        default="/mlcv2025/Datasets/HCMAI25/batch2/video",
        help="Folder containing source videos",
    )
    parser.add_argument(
        "--output",
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--existing",
        default="",
        help="Existing CSV to skip already-probed video_id (and merge unless --no_merge)",
    )
    parser.add_argument("--video_list", default=None, help="Optional text file of video names/ids")
    parser.add_argument("--workers", type=int, default=8, help="Parallel probe workers")
    parser.add_argument("--recursive", action="store_true", help="Scan input_dir recursively")
    parser.add_argument("--no_merge", action="store_true", help="Write only newly probed rows")
    parser.add_argument("--force", action="store_true", help="Re-probe videos already in --existing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    existing_path = args.existing or (args.output if os.path.isfile(args.output) else "")
    existing_ids = set() if args.force else load_existing_ids(existing_path)
    allow = load_video_list(args.video_list)

    files = list_video_files(input_dir, args.recursive)
    pending = []
    for path in files:
        name = os.path.basename(path)
        vid = os.path.splitext(name)[0]
        if allow is not None and vid not in allow and name not in allow:
            continue
        if vid in existing_ids:
            continue
        pending.append(path)

    print(f"Found {len(files)} videos in {input_dir}")
    print(f"Already in metadata: {len(existing_ids)}")
    print(f"Pending probe: {len(pending)}")

    new_rows = []
    failed = []
    if pending:
        workers = max(1, args.workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(probe_video, p): p for p in pending}
            done = 0
            for fut in as_completed(futures):
                done += 1
                path = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:
                    failed.append((path, str(exc)))
                    continue
                if not row or row.get("error"):
                    failed.append((path, row.get("error") if row else "empty"))
                    continue
                new_rows.append(row)
                if done % 25 == 0 or done == len(pending):
                    print(f"  probed {done}/{len(pending)}")

    if args.no_merge:
        rows = new_rows
    else:
        rows = merge_existing(existing_path, new_rows)

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows -> {args.output}")
    print(f"Newly probed: {len(new_rows)} | failed: {len(failed)}")
    if failed:
        for path, err in failed[:20]:
            print(f"  FAIL {os.path.basename(path)}: {err}", file=sys.stderr)
        if len(failed) > 20:
            print(f"  ... {len(failed) - 20} more failures", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
