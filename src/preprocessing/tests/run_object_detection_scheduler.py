#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Schedule object detection jobs across two GPUs using run_pipeline.py stage object_detection."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory where metadata and keyframes live.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="Comma-separated GPU IDs to use, or auto to pick the best GPUs.",
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=2,
        help="Number of GPUs to select when using auto.",
    )
    parser.add_argument(
        "--job_video_count",
        type=int,
        default=1,
        help="Number of videos per scheduled object detection job.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="Metadata format for keyframes and object detection output.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "demo"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--video_list",
        type=str,
        default=None,
        help="Optional text file listing selected video filenames or full paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun object detection stage even if pipeline log says it completed.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to launch run_pipeline.py.",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=5.0,
        help="Seconds between polling GPU job completion.",
    )
    return parser.parse_args()


def query_gpus():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        idx, used, total, util = parts
        try:
            used = int(used)
            total = int(total)
            util = int(util)
        except ValueError:
            continue
        free = total - used
        gpus.append({"id": idx, "free_mem": free, "util": util})
    return gpus


def select_gpus(gpus_arg, gpu_count):
    if gpus_arg.lower() == "auto":
        gpus = query_gpus()
        if not gpus:
            raise RuntimeError("Cannot auto-select GPUs because nvidia-smi failed.")
        gpus.sort(key=lambda x: (-x["free_mem"], x["util"]))
        selected = [gpu["id"] for gpu in gpus[:gpu_count]]
        print("Selected GPUs:")
        for gpu in gpus[:gpu_count]:
            print(f"  GPU {gpu['id']} free_mem={gpu['free_mem']}MiB util={gpu['util']}%")
        return selected
    ids = [s.strip() for s in gpus_arg.split(",") if s.strip()]
    if not ids:
        raise ValueError("No GPU IDs provided.")
    return ids[:gpu_count]


def read_video_list(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video list not found: {path}")
    videos = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            videos.append(os.path.basename(text))
    return videos


def video_ids_from_keyframe_metadata(metadata_dir, fmt):
    files = []
    for suffix in [fmt, "json"]:
        files.extend(Path(metadata_dir).glob(f"*_keyframes.{suffix}"))
    ids = set()
    for path in files:
        name = path.name
        if name.endswith(f"_keyframes.{path.suffix.lstrip('.')}"):
            ids.add(name.rsplit("_keyframes.", 1)[0])
    return sorted(ids)


def input_video_map(input_dir):
    extension_map = {}
    for file_name in sorted(os.listdir(input_dir)):
        lower = file_name.lower()
        if lower.endswith((".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm")):
            video_id = os.path.splitext(file_name)[0]
            if video_id not in extension_map:
                extension_map[video_id] = file_name
    return extension_map


def video_done(output_dir, video_id, fmt):
    objects_dir = Path(output_dir) / "metadata" / "objects"
    for suffix in [fmt, "json"]:
        if (objects_dir / f"{video_id}_objects.{suffix}").exists():
            return True
    return False


def build_command(args, batch_file):
    script_dir = Path(__file__).resolve().parent
    cmd = [
        args.python,
        str(script_dir / "run_pipeline.py"),
        "--stage",
        "object_detection",
        "--mode",
        args.mode,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--video_list",
        str(batch_file),
        "--format",
        args.format,
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def write_job_file(job_dir, job_idx, video_ids):
    job_path = job_dir / f"object_detection_job_{job_idx:03d}.txt"
    with job_path.open("w", encoding="utf-8") as handle:
        for vid in video_ids:
            handle.write(f"{vid}.mp4\n")
    return job_path


def main():
    args = parse_args()
    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    gpus = select_gpus(args.gpus, args.gpu_count)
    print(f"Using GPUs: {', '.join(gpus)}")

    metadata_keyframes_dir = Path(args.output_dir) / "metadata" / "keyframes"
    if not metadata_keyframes_dir.exists():
        raise FileNotFoundError(f"Keyframe metadata directory not found: {metadata_keyframes_dir}")

    if args.video_list:
        selected = set(read_video_list(args.video_list))
    else:
        selected = None

    candidate_ids = video_ids_from_keyframe_metadata(metadata_keyframes_dir, args.format)
    if selected is not None:
        candidate_ids = [vid for vid in candidate_ids if f"{vid}.mp4" in selected or vid in selected]

    pending_ids = [vid for vid in candidate_ids if not video_done(args.output_dir, vid, args.format)]
    print(f"Found {len(candidate_ids)} videos with keyframe metadata, {len(pending_ids)} pending object detection jobs.")
    if not pending_ids:
        print("No pending object detection jobs.")
        return

    job_dir = Path(args.output_dir) / "object_detection_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)

    running = {}
    job_idx = 0

    try:
        while pending_ids or running:
            # Refresh pending list in case files are generated externally
            all_candidate_ids = video_ids_from_keyframe_metadata(metadata_keyframes_dir, args.format)
            if selected is not None:
                all_candidate_ids = [vid for vid in all_candidate_ids if f"{vid}.mp4" in selected or vid in selected]
            pending_ids = [vid for vid in all_candidate_ids if not video_done(args.output_dir, vid, args.format)]

            # clean completed jobs
            for gpu_id, info in list(running.items()):
                proc = info["proc"]
                if proc.poll() is not None:
                    print(f"GPU {gpu_id} finished job {info['job_idx']} videos={info['video_ids']} exit={proc.returncode}")
                    del running[gpu_id]

            available_gpus = [gpu for gpu in gpus if gpu not in running]
            while available_gpus and pending_ids:
                gpu_id = available_gpus.pop(0)
                batch_ids = [pending_ids.pop(0) for _ in range(min(args.job_video_count, len(pending_ids)))]
                job_file = write_job_file(job_dir, job_idx, batch_ids)
                cmd = build_command(args, job_file)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                print(f"Starting job {job_idx} on GPU {gpu_id}: {batch_ids}")
                proc = subprocess.Popen(cmd, env=env)
                running[gpu_id] = {"proc": proc, "job_idx": job_idx, "video_ids": batch_ids}
                job_idx += 1

            if running:
                time.sleep(args.poll_interval)
            elif not pending_ids:
                break

        print("All object detection jobs complete.")
    finally:
        print(f"Job files are saved in: {job_dir}")


if __name__ == "__main__":
    main()
