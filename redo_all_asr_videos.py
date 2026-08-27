#!/usr/bin/env python3
"""
Redo (overwrite) ASR metadata for ALL keyframes of SPECIFIED videos.

Unlike run_missing_asr.py which only updates a "missing" subset of keyframes,
this script re-generates ASR for EVERY keyframe of a given video list (i.e.
makes a fresh, complete *_asr.jsonl from scratch, overwriting the old one).

IMPORTANT: Run this script with the SAME interpreter used for run_missing_asr.py.
           Do NOT run with the local .venv (Python 3.14, broken numpy); use the
           aic2026_backend conda env (Python 3.10) or its absolute python path.

Usage examples:
  # Auto - redo for the 114 GROUP1 videos in corrupt report
  python redo_all_asr_videos.py --gpu 0

  # Manual list
  python redo_all_asr_videos.py --gpu 0 --video_ids L23_V018 L24_V005 L26_V008
  python redo_all_asr_videos.py --gpu 1 --video_list my_videos.txt
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# CUDA library and Environment variable resolution
#   (copy-pasted from run_missing_asr.py, ZERO additions / removals)
# ---------------------------------------------------------------------------
_CUDA_LIB_SEARCH_ROOTS = [
    "/usr/local/cuda/lib64",
    "/usr/local/lib/ollama/cuda_v12",
    "/usr/local/lib/ollama/cuda_v13",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
]

def _get_venv_nvidia_lib_dirs():
    dirs = []
    try:
        import site
        site_pkgs = site.getsitepackages() if hasattr(site, "getsitepackages") else []
        import sysconfig
        sp = sysconfig.get_path("purelib")
        if sp:
            site_pkgs = list(site_pkgs) + [sp]
        for sp in site_pkgs:
            for lib_dir in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
                dirs.append(lib_dir)
    except Exception:
        pass
    return dirs

def resolve_cuda_lib_path():
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
        matches = glob.glob(os.path.join(directory, "libcublas.so*"))
        if matches:
            found_dirs.append(directory)
    return ":".join(found_dirs)

# Pre-exec step to restart with correct LD_LIBRARY_PATH if not set
if __name__ == "__main__" and not os.environ.get("_LD_LIBRARY_PATH_SET"):
    cuda_path = resolve_cuda_lib_path()
    if cuda_path:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = f"{cuda_path}:{existing}" if existing else cuda_path
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_LD_LIBRARY_PATH_SET"] = "1"
        # Restart the script with correct environment
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Re-generate ASR from scratch for ALL keyframes of a set of videos."
    )
    p.add_argument(
        "--video_ids",
        type=str,
        nargs="*",
        default=None,
        help="Explicit video ID list (e.g. --video_ids L23_V018 L24_V005).",
    )
    p.add_argument(
        "--video_list",
        type=str,
        default=None,
        help="Txt file with one video_id per line (# comment allowed).",
    )
    p.add_argument(
        "--auto_corrupt_group1",
        action="store_true",
        default=True,
        help="If no video_ids/list given -> auto load the GROUP1 videos from corrupt report. Default True.",
    )
    p.add_argument(
        "--corrupt_report",
        type=str,
        default="/AIClub_NAS/core_baotg/phong/AIC_2026/corrupt_keyframes_list.txt",
    )
    p.add_argument(
        "--video_dirs",
        type=str,
        nargs="+",
        default=["/mlcv1/Datasets/HCMAI25/full", "/mlcv2025/Datasets/HCMAI25/batch2/video"],
    )
    p.add_argument(
        "--keyframes_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes",
    )
    p.add_argument(
        "--asr_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr",
    )
    p.add_argument(
        "--backup_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr_backup_before_redo",
        help="Backup old *_asr.jsonl here BEFORE overwriting (per video).",
    )
    p.add_argument("--gpu", type=str, default="0", help="GPU id, e.g. 0,1 or CPU empty.")
    p.add_argument(
        "--asr_model_size",
        type=str,
        default="medium",
        choices=[
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en", "distil-small.en",
            "medium", "medium.en", "distil-medium.en",
            "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo",
        ],
        help="faster-whisper model size. Note: 'large' -> alias to large-v3. "
             "large-v3-turbo is smaller than large-v3 (faster, less VRAM) with near-same WER.",
    )
    p.add_argument("--asr_language", type=str, default="vi")
    p.add_argument(
        "--asr_compute_type",
        type=str,
        default="",
        choices=["", "float16", "float32", "int8", "int8_float16", "int16", "bfloat16"],
        help="faster-whisper compute type: empty = auto (float16 if GPU else int8). "
             "Medium on GPU needs ~5.3GB for float16, ~3.2GB for int8_float16.",
    )
    p.add_argument("--window_sec", type=float, default=3.0, help="Half-window around kf timestamp.")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print plan (what videos, counts, expected paths, check video file exists); no actual transcribe.",
    )
    p.add_argument(
        "--skip_existing_check",
        action="store_true",
        help="Do not skip even if asr file exists (always redo). Default: always redo for list anyway.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers (mostly same as run_missing_asr.py, key change: load ALL keyframes)
# ---------------------------------------------------------------------------
def load_video_list(args):
    vids = []
    seen = set()

    def _add(vid):
        vid = (vid or "").strip()
        if not vid or vid.startswith("#") or vid in seen:
            return
        seen.add(vid)
        vids.append(vid)

    if args.video_ids:
        for v in args.video_ids:
            _add(v)
    if args.video_list and os.path.exists(args.video_list):
        with open(args.video_list, "r", encoding="utf-8") as f:
            for line in f:
                _add(line)

    if not vids and args.auto_corrupt_group1 and os.path.exists(args.corrupt_report):
        # Parse GROUP1 video list from the corrupt report "chi tiết từng video" section
        # marker: [GROUP1_SINGLE_TEXT_LOOP] Video: XXXXX
        with open(args.corrupt_report, "r", encoding="utf-8") as f:
            for line in f:
                if "[GROUP1_SINGLE_TEXT_LOOP]" in line:
                    part = line.split("Video:", 1)[-1].split("File:", 1)[0].strip()
                    _add(part)

    return vids


def find_video_path(video_id, video_dirs):
    for ext in [".mp4", ".mkv", ".avi"]:
        for d in video_dirs:
            p = os.path.join(d, f"{video_id}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_ALL_keyframes(video_id, keyframes_metadata_dir):
    """Load ALL keyframes for this video, not a subset (main change from original)."""
    csv_path = os.path.join(keyframes_metadata_dir, f"{video_id}_keyframes.csv")
    json_path = os.path.join(keyframes_metadata_dir, f"{video_id}_keyframes.json")
    data = {}

    def _norm(row):
        try:
            row["frame_idx"] = int(row["frame_idx"])
        except Exception:
            row["frame_idx"] = 0
        try:
            row["pts_time"] = float(row["pts_time"])
        except Exception:
            row["pts_time"] = 0.0
        try:
            row["timestamp"] = int(row["timestamp"])
        except Exception:
            row["timestamp"] = int(round(row["pts_time"] * 1000))
        return row

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kf_id = row.get("keyframe_id", "").strip()
                if not kf_id:
                    continue
                if "video_id" not in row or not row["video_id"]:
                    row["video_id"] = video_id
                data[kf_id] = _norm(row)

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except Exception:
                records = []
        for row in records:
            kf_id = row.get("keyframe_id", "").strip()
            if not kf_id or kf_id in data:
                continue
            if "video_id" not in row or not row["video_id"]:
                row["video_id"] = video_id
            data[kf_id] = _norm(row)

    return data


def map_asr_to_keyframe(row, raw_segments, window_sec):
    try:
        t_kf = float(row.get("pts_time", 0.0))
    except Exception:
        t_kf = 0.0
    w_start = max(0.0, t_kf - window_sec)
    w_end = t_kf + window_sec
    matches = []
    for seg in raw_segments:
        try:
            seg_start = float(seg["segment_start"])
            seg_end = float(seg["segment_end"])
        except Exception:
            continue
        if seg_end > w_start and seg_start < w_end:
            matches.append(seg)
    return {
        **row,
        "asr_metadata": {
            "window_start": w_start,
            "window_end": w_end,
            "asr_text_6s": " ".join([seg["text"] for seg in matches]).strip() if matches else "",
            "has_speech": len(matches) > 0,
            "raw_segments": matches,
        },
    }


def backup_old_file(video_id, asr_metadata_dir, backup_dir):
    src = os.path.join(asr_metadata_dir, f"{video_id}_asr.jsonl")
    if not os.path.exists(src):
        return None
    os.makedirs(backup_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(backup_dir, f"{video_id}_asr.{ts}.jsonl.bak")
    import shutil
    shutil.copy2(src, dst)
    return dst


def save_asr_records(video_id, records, asr_metadata_dir):
    os.makedirs(asr_metadata_dir, exist_ok=True)
    out_path = os.path.join(asr_metadata_dir, f"{video_id}_asr.jsonl")
    def _fi(rec):
        try:
            return int(rec.get("frame_idx", 0))
        except Exception:
            return 0
    records = sorted(records, key=_fi)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path, len(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    video_ids = load_video_list(args)
    if not video_ids:
        print("[-] No videos to process. Provide --video_ids / --video_list or valid --corrupt_report GROUP1 list.")
        sys.exit(2)

    print(f"[PLAN] Will redo ASR for {len(video_ids)} videos.")
    print(f"       GPU={args.gpu}   model={args.asr_model_size}   lang={args.asr_language}   win±{args.window_sec}s")
    print(f"       keyframes_dir={args.keyframes_metadata_dir}")
    print(f"       asr_dir (overwrite)={args.asr_metadata_dir}")
    print(f"       backup_dir={args.backup_dir}")
    print(f"       video_dirs={args.video_dirs}")

    # Dry-run check: list video paths + kf counts
    plan = []
    for vid in video_ids:
        vp = find_video_path(vid, args.video_dirs)
        kfs = load_ALL_keyframes(vid, args.keyframes_metadata_dir)
        old_asr_exists = os.path.exists(os.path.join(args.asr_metadata_dir, f"{vid}_asr.jsonl"))
        plan.append({
            "video_id": vid,
            "video_path": vp,
            "num_kf": len(kfs),
            "old_asr_exists": old_asr_exists,
        })

    missing_video = [x for x in plan if not x["video_path"]]
    no_kf = [x for x in plan if x["num_kf"] == 0]

    print(f"\n[DRY_SCAN SUMMARY] videos with missing mp4: {len(missing_video)}, with 0 keyframes: {len(no_kf)}")
    if missing_video:
        for x in missing_video[:20]:
            print(f"   ! NOT FOUND VIDEO FILE: {x['video_id']}")
    if no_kf:
        for x in no_kf[:20]:
            print(f"   ! 0 KEYFRAMES LOADED: {x['video_id']}")

    total_kf_plan = sum(x["num_kf"] for x in plan if x["video_path"])
    print(f"   Videos OK to redo: {len(plan) - len(missing_video) - len(no_kf)}  (total keyframes: {total_kf_plan})")

    if args.dry_run:
        print("\n[DRY_RUN] DONE - no transcription ran. Remove --dry_run to execute.")
        return

    # ----- REAL RUN -----
    import torch
    from faster_whisper import WhisperModel

    print(f"\n[INIT] WhisperModel size={args.asr_model_size} on GPU={args.gpu}...")
    cuda_available = torch.cuda.is_available()
    if args.asr_compute_type:
        compute_type = args.asr_compute_type
    else:
        compute_type = "float16" if cuda_available else "int8"
    device = "cuda" if cuda_available else "cpu"
    print(f"       torch={torch.__version__}  cuda_ok={cuda_available}  device={device}  dtype={compute_type}")
    model = WhisperModel(
        args.asr_model_size, device=device, compute_type=compute_type, num_workers=1
    )
    print("[INIT] Model loaded.\n")

    run_log = []
    t_start_all = time.perf_counter()
    processed_vids = 0

    for idx, vid in enumerate(video_ids, 1):
        info = next((p for p in plan if p["video_id"] == vid), None)
        prefix = f"[{idx}/{len(video_ids)}] {vid}"
        if not info or not info["video_path"] or info["num_kf"] == 0:
            print(f"{prefix} -> SKIP (video missing or 0 keyframes loaded)")
            run_log.append({"video_id": vid, "status": "skip", "reason": "no_video_or_no_kf"})
            continue

        kf_all = load_ALL_keyframes(vid, args.keyframes_metadata_dir)
        print(f"\n{prefix}  kf={len(kf_all)}  file={os.path.basename(info['video_path'])} ({os.path.getsize(info['video_path'])/1e9:.2f} GB)")

        # Backup old asr first
        bkp = backup_old_file(vid, args.asr_metadata_dir, args.backup_dir)
        if bkp:
            print(f"  backup: {bkp}")

        # Whisper transcribe
        t0 = time.perf_counter()
        try:
            segments, _ = model.transcribe(
                info["video_path"],
                language=args.asr_language,
                beam_size=args.beam_size,
            )
            raw_segments = [
                {
                    "segment_start": float(s.start),
                    "segment_end": float(s.end),
                    "text": s.text.strip(),
                }
                for s in segments
            ]
            t_trans = time.perf_counter() - t0
            n_speech = sum(1 for s in raw_segments if s["text"])
            print(f"  whisper ok: {len(raw_segments)} segments (with text: {n_speech}) in {t_trans:.1f}s")
        except Exception as e:
            print(f"  WHISPER FAILED: {e}")
            run_log.append({"video_id": vid, "status": "error", "error": f"transcribe:{e}"})
            continue

        # Map each KF -> ASR. IMPORTANT: do for ALL kf_ids (not a subset).
        mapped = {}
        for kf_id, row in kf_all.items():
            mapped[kf_id] = map_asr_to_keyframe(row, raw_segments, args.window_sec)

        records = list(mapped.values())
        # Sanity stats before write
        n_has_speech = sum(1 for r in records if r["asr_metadata"].get("has_speech"))
        unique_texts = set(r["asr_metadata"].get("asr_text_6s", "") for r in records)
        unique_non_empty = set(t for t in unique_texts if t)
        print(f"  mapped: {len(records)} recs  has_speech={n_has_speech}  unique_asr_texts={len(unique_non_empty)}")

        out_path, n_written = save_asr_records(vid, records, args.asr_metadata_dir)
        dur = time.perf_counter() - t0
        processed_vids += 1
        print(f"  SAVED -> {out_path} ({n_written} records) in {dur:.1f}s total\n")

        run_log.append({
            "video_id": vid,
            "status": "ok",
            "kf_count": len(records),
            "has_speech": n_has_speech,
            "unique_non_empty_asr": len(unique_non_empty),
            "whisper_segments": len(raw_segments),
            "elapsed_sec": round(dur, 2),
            "backup": bkp,
            "output": out_path,
        })

    total_elapsed = time.perf_counter() - t_start_all
    status_counts = {}
    for x in run_log:
        s = x["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print("\n" + "=" * 90)
    print(f"[ALL DONE] processed={processed_vids}/{len(video_ids)} in {total_elapsed:.1f}s")
    print(f"         status breakdown: {status_counts}")
    print(f"         model: {args.asr_model_size}  device: {device}")
    print("=" * 90)

    # Save log
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"redo_asr_log_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    log_obj = {
        "args": {
            k: v for k, v in vars(args).items() if k not in ("corrupt_report",)
        },
        "summary": {
            "n_videos_in": len(video_ids),
            "n_videos_ok": status_counts.get("ok", 0),
            "n_videos_skip": status_counts.get("skip", 0),
            "n_videos_error": status_counts.get("error", 0),
            "elapsed_total_sec": round(total_elapsed, 2),
            "device": device,
            "model_size": args.asr_model_size,
        },
        "per_video": run_log,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_obj, f, ensure_ascii=False, indent=2)
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
