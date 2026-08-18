#!/usr/bin/env python3
import argparse
import glob
import math
import os
import subprocess
import sys
import threading
import time
from itertools import islice, cycle
from pathlib import Path


# ---------------------------------------------------------------------------
# CUDA library resolution
# ---------------------------------------------------------------------------
_CUDA_LIB_SEARCH_ROOTS = [
    "/usr/local/cuda/lib64",
    "/usr/local/lib/ollama/cuda_v12",
    "/usr/local/lib/ollama/cuda_v13",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
]


def _get_venv_nvidia_lib_dirs():
    """Find nvidia CUDA lib dirs installed as Python wheels inside the current venv."""
    dirs = []
    try:
        import site
        site_pkgs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        # Also check the running interpreter's site-packages
        import sysconfig
        sp = sysconfig.get_path("purelib")
        if sp:
            site_pkgs = list(site_pkgs) + [sp]
        for sp in site_pkgs:
            # nvidia/<cu_version>/lib/ layout used by nvidia-cublas-cu12 / cu13 wheels
            for lib_dir in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
                dirs.append(lib_dir)
    except Exception:
        pass
    return dirs


def resolve_cuda_lib_path():
    """Return a colon-separated list of directories that contain CUDA shared
    libraries required by faster-whisper / CTranslate2 (libcublas.so.*).
    Searches common system paths, ollama paths, and nvidia wheel paths inside
    the active venv. The result can be safely prepended to LD_LIBRARY_PATH.
    """
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    search_dirs = (
        _get_venv_nvidia_lib_dirs()
        + _CUDA_LIB_SEARCH_ROOTS
        + [p for p in existing.split(":") if p]
    )
    found_dirs = []
    seen = set()
    for directory in search_dirs:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        # Accept any version of libcublas shared library
        matches = glob.glob(os.path.join(directory, "libcublas.so*"))
        if matches:
            found_dirs.append(directory)
    return ":".join(found_dirs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipeline in parallel video batches and choose the best available GPUs."
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
        help="Output directory for pipeline results.",
    )
    parser.add_argument(
        "--video_list",
        type=str,
        required=True,
        help="Text file listing selected videos, one filename or full path per line.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="Comma-separated GPU device IDs to use, or auto to detect best GPUs.",
    )
    parser.add_argument(
        "--gpu_count",
        type=int,
        default=1,
        help="When --gpus=auto, select this many best GPUs. For object detection this is typically 1, and auto mode picks the GPUs with the most free memory first.",
    )
    parser.add_argument(
        "--min_free_mem",
        type=int,
        default=3000,
        help="Minimum free GPU memory (MiB) required for auto GPU selection. Use 6000 MiB for captioning with OpenGVLab/InternVL2_5-2B INT4; otherwise the scheduler may pick GPUs that are too tight for model load.",
    )
    parser.add_argument(
        "--max_gpu_util",
        type=int,
        default=70,
        help="Maximum GPU utilization percent for auto GPU selection.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of videos per worker batch.",
    )
    parser.add_argument(
        "--batch_count",
        type=int,
        default=None,
        help="Number of video batches to create. Overrides batch_size if set.",
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=1,
        help="Start dispatching from this batch number (1-based). Useful for resuming after a specific batch.",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=None,
        help="Maximum concurrent batches. Defaults to number of selected GPUs.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="shot_cut",
        choices=["shot_cut", "keyframe_ext", "asr_whisper", "object_detection", "captioning", "all", "stage12"],
        help="Pipeline stage to run for each batch. Temporary override: stage12 runs shot_cut + keyframe_ext sequentially in the worker process.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "demo"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="Metadata output format.",
    )
    parser.add_argument(
        "--asr_model_size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="faster-whisper model size for ASR stage.",
    )
    parser.add_argument(
        "--asr_language",
        type=str,
        default="vi",
        help="Language code for ASR (e.g. 'vi' for Vietnamese, 'en' for English).",
    )
    parser.add_argument(
        "--caption_model_id",
        type=str,
        default="OpenGVLab/InternVL2_5-2B",
        help="Captioning model ID. Recommended default for < 6 GB free VRAM is OpenGVLab/InternVL2_5-2B with INT4 quantization.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun each stage even if pipeline log indicates completion.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for subprocesses.",
    )
    return parser.parse_args()


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
            videos.append(text)
    return videos


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
        index, mem_used, mem_total, util = parts
        try:
            mem_used = int(mem_used)
            mem_total = int(mem_total)
            util = int(util)
            free_mem = mem_total - mem_used
        except ValueError:
            continue
        gpus.append(
            {
                "id": index,
                "free_mem": free_mem,
                "util": util,
                "used_mem": mem_used,
                "total_mem": mem_total,
            }
        )
    return gpus


def select_gpus(gpus_arg, gpu_count, min_free_mem, max_gpu_util):
    if gpus_arg.lower() == "auto":
        gpus = query_gpus()
        if not gpus:
            raise RuntimeError(
                "Unable to detect GPUs automatically. Please specify --gpus explicitly."
            )
        available = [
            gpu
            for gpu in gpus
            if gpu["free_mem"] >= min_free_mem and gpu["util"] <= max_gpu_util
        ]
        if not available:
            raise RuntimeError(
                f"No GPUs currently meet the minimum free memory requirement ({min_free_mem} MiB). "
                "Reduce the workload or rerun when more GPU memory is available."
            )
        available.sort(key=lambda gpu: (-gpu["free_mem"], gpu["util"]))
        if gpu_count is not None:
            available = available[:gpu_count]
        return [gpu["id"] for gpu in available]

    ids = [gpu.strip() for gpu in gpus_arg.split(",") if gpu.strip()]
    if not ids:
        raise ValueError("At least one GPU id is required via --gpus.")
    return ids


def wait_for_captioning_gpus(gpu_count_target=2, min_free_mem=8000, max_gpu_util=70, poll_interval=5):
    """For captioning, keep scanning GPU memory until at least two GPUs are free enough.

    This is intentionally stage-specific: captioning loads heavy VLM weights, so we wait until
    we have two GPUs with roughly 8 GiB free memory before launching. Once the target count is met,
    we stop scanning and proceed immediately.
    """
    print(
        f"Captioning scheduler: waiting for at least {gpu_count_target} GPU(s) with "
        f"{min_free_mem} MiB free memory and utilization <= {max_gpu_util}%..."
    )
    while True:
        gpus = query_gpus()
        if not gpus:
            print("No GPUs detected yet; waiting for CUDA to become available...")
            time.sleep(poll_interval)
            continue

        available = [
            gpu
            for gpu in gpus
            if gpu["free_mem"] >= min_free_mem and gpu["util"] <= max_gpu_util
        ]
        if len(available) >= gpu_count_target:
            available.sort(key=lambda gpu: (-gpu["free_mem"], gpu["util"]))
            selected = [gpu["id"] for gpu in available[:gpu_count_target]]
            print(
                "Captioning ready: selected GPUs "
                + ", ".join(str(gpu_id) for gpu_id in selected)
                + " with sufficient free memory."
            )
            return selected

        best = sorted(gpus, key=lambda gpu: (-gpu["free_mem"], gpu["util"]))
        top = best[:5]
        details = ", ".join(
            f"GPU{gpu['id']} free={gpu['free_mem']}MiB util={gpu['util']}%" for gpu in top
        )
        print(f"Not enough captioning-ready GPUs yet. Current candidates: {details}. Retrying in {poll_interval}s...")
        time.sleep(poll_interval)


def chunked(items, size):
    it = iter(items)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch


def write_batch_list(base_dir, batch_idx, batch_entries):
    batch_file = base_dir / f"video_list_batch_{batch_idx:03d}.txt"
    with batch_file.open("w", encoding="utf-8") as handle:
        for entry in batch_entries:
            handle.write(f"{entry}\n")
    return batch_file


def build_command(args, batch_file):
    stage = args.stage
    if stage == "stage12":
        py = args.python
        stage_cmd = (
            f'"{py}" run_pipeline.py --mode {args.mode} --stage shot_cut --input_dir "{args.input_dir}" '
            f'--output_dir "{args.output_dir}" --video_list "{batch_file}" --format {args.format} '
            f'{"--force" if args.force else ""} && '
            f'"{py}" run_pipeline.py --mode {args.mode} --stage keyframe_ext --input_dir "{args.input_dir}" '
            f'--output_dir "{args.output_dir}" --video_list "{batch_file}" --format {args.format} '
            f'{"--force" if args.force else ""}'
        ).strip()
        return ["bash", "-lc", stage_cmd]

    cmd = [
        args.python,
        "run_pipeline.py",
        "--mode",
        args.mode,
        "--stage",
        stage,
        "--input_dir",
        args.input_dir,
        "--output_dir",
        args.output_dir,
        "--video_list",
        str(batch_file),
        "--format",
        args.format,
    ]
    if getattr(args, "asr_model_size", None):
        cmd.extend(["--asr_model_size", args.asr_model_size])
    if getattr(args, "asr_language", None):
        cmd.extend(["--asr_language", args.asr_language])
    if getattr(args, "caption_model_id", None):
        cmd.extend(["--caption_model_id", args.caption_model_id])
    if args.force:
        cmd.append("--force")
    return cmd


def stream_proc_output(proc, gpu, batch_idx, log_file):
    with open(log_file, "a", encoding="utf-8") as handle:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                break
            text = line.rstrip()
            prefix = f"[batch {batch_idx+1} | GPU {gpu}] {text}"
            handle.write(prefix + "\n")
            handle.flush()
            print(prefix)


def wait_for_slot(procs, max_parallel):
    while len(procs) >= max_parallel:
        for batch_idx, batch_file, proc, gpu, reader in list(procs):
            ret = proc.poll()
            if ret is not None:
                if reader and reader.is_alive():
                    reader.join(timeout=1)
                print(f"[batch {batch_idx+1} | GPU {gpu}] finished with exit code {ret}.")
                procs.remove((batch_idx, batch_file, proc, gpu, reader))
        if len(procs) >= max_parallel:
            time.sleep(2)


def launch_batch(args, batch_dir, batch_log_dir, idx, batch, gpu, procs):
    """Write batch list, build command, and spawn the worker subprocess."""
    batch_file = write_batch_list(batch_dir, idx, batch)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cuda_lib_path = resolve_cuda_lib_path()
    if cuda_lib_path:
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{cuda_lib_path}:{existing_ld}" if existing_ld else cuda_lib_path
        )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = build_command(args, batch_file)
    log_file = batch_log_dir / f"batch_{idx+1:03d}_gpu_{gpu}.log"
    print(f"Launching batch {idx+1} on GPU {gpu}: {batch_file.name}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader = threading.Thread(
        target=stream_proc_output,
        args=(proc, gpu, idx, log_file),
        daemon=True,
    )
    reader.start()
    procs.append((idx, batch_file, proc, gpu, reader))
    return batch_file


def iter_batches(args):
    """Generator that reloads the video list from disk on every call.

    Each time a new batch needs to be dispatched the file at ``args.video_list``
    is re-read so that additions or removals made while the script is running
    are reflected in subsequent batches without restarting the process.
    """
    videos = read_video_list(args.video_list)
    if not videos:
        raise ValueError("Video list is empty.")

    if args.batch_count is not None and args.batch_count > 0:
        batch_size = math.ceil(len(videos) / args.batch_count)
    else:
        batch_size = args.batch_size

    yield from chunked(videos, batch_size)


def main():
    args = parse_args()

    # Captioning stage is special: keep scanning until two GPUs are clean enough to run the VLM.
    if args.stage == "captioning" and args.gpus.lower() == "auto":
        if args.gpu_count is None:
            args.gpu_count = 2
        if args.min_free_mem < 8000:
            print(f"Auto-adjusting --min_free_mem to 8000 MiB for {args.caption_model_id} captioning.")
            args.min_free_mem = 8000
        gpus = wait_for_captioning_gpus(
            gpu_count_target=args.gpu_count,
            min_free_mem=args.min_free_mem,
            max_gpu_util=args.max_gpu_util,
            poll_interval=5,
        )
    else:
        gpus = select_gpus(args.gpus, args.gpu_count, args.min_free_mem, args.max_gpu_util)

    if not gpus:
        raise ValueError("No GPUs available for batch execution.")

    # ------------------------------------------------------------------
    # Read video list once upfront just to print the summary and compute
    # the total batch count; the actual per-batch reload happens inside
    # iter_batches() right before each batch is dispatched.
    # ------------------------------------------------------------------
    videos_initial = read_video_list(args.video_list)
    if not videos_initial:
        raise ValueError("Video list is empty.")

    if args.batch_count is not None and args.batch_count > 0:
        batch_size_preview = math.ceil(len(videos_initial) / args.batch_count)
    else:
        batch_size_preview = args.batch_size
    total_batches_preview = math.ceil(len(videos_initial) / batch_size_preview)

    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpus)
    max_parallel = min(max_parallel, len(gpus))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = output_dir / "batch_lists"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_log_dir = output_dir / "batch_logs"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "stage12":
        print(
            "Temporary mode: stage12 -> only shot_cut + keyframe_ext will be executed for each batch."
        )
    if args.start_batch and args.start_batch > 1:
        print(f"Resuming from batch {args.start_batch}.")
    print(
        f"Found {len(videos_initial)} videos (from {args.video_list}), "
        f"~{total_batches_preview} batches with "
        f"{len(gpus)} GPU(s) selected and max {max_parallel} concurrent batches."
    )
    print("[NOTE] video_list is reloaded from disk before each batch is dispatched.")

    procs = []
    batch_files = []
    gpu_cycle = cycle(gpus)

    try:
        for idx, batch in enumerate(iter_batches(args)):
            if args.start_batch and idx + 1 < args.start_batch:
                continue
            wait_for_slot(procs, max_parallel)
            gpu = next(gpu_cycle)
            batch_file = launch_batch(args, batch_dir, batch_log_dir, idx, batch, gpu, procs)
            batch_files.append(batch_file)

        for batch_idx, batch_file, proc, gpu, reader in procs:
            ret = proc.wait()
            if reader and reader.is_alive():
                reader.join(timeout=1)
            print(f"[batch {batch_idx+1} | GPU {gpu}] finished with exit code {ret}.")
    finally:
        if batch_files:
            print(f"Batch list files are kept in {batch_dir}")
            print(f"Batch log files are kept in {batch_log_dir}")


if __name__ == "__main__":
    main()
