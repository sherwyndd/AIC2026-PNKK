#!/usr/bin/env python3
"""
Redo (overwrite) ASR metadata for ALL keyframes of ALL videos.

Scans ALL *_keyframes.csv in --keyframes_metadata_dir automatically —
no need to specify video lists. Overwrites existing *_asr.jsonl in place
(with optional backup).

IMPORTANT: Use the aic2026_backend conda env (Python 3.10), NOT .venv (broken numpy).

Usage:
  # Run hết tất cả video, model large-v3-turbo float16 (manh nhat < 10GB VRAM)
  python redo_all_asr_videos.py --gpu 0 --asr_model_size large-v3-turbo --asr_compute_type float16

  # large-v2 float16 (chat luong cao nhat, ~8-9GB VRAM)
  python redo_all_asr_videos.py --gpu 0 --asr_model_size large-v2 --asr_compute_type float16

  # Dry-run xem plan (khong transcribe)
  python redo_all_asr_videos.py --dry_run

  # Bo qua video da co asr roi (chi lam video chua co)
  python redo_all_asr_videos.py --gpu 0 --asr_model_size large-v3-turbo --asr_compute_type float16 --skip_existing

  # Resume tu video cu the (theo thu tu alpha, bat dau tu L24_V001)
  python redo_all_asr_videos.py --gpu 0 --asr_model_size large-v3-turbo --asr_compute_type float16 --resume_from L24_V001
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# CUDA library path resolution (copy from run_missing_asr.py)
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


# Pre-exec: restart with correct LD_LIBRARY_PATH if not already set
if __name__ == "__main__" and not os.environ.get("_LD_LIBRARY_PATH_SET"):
    cuda_path = resolve_cuda_lib_path()
    if cuda_path:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = f"{cuda_path}:{existing}" if existing else cuda_path
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_LD_LIBRARY_PATH_SET"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Re-generate ASR for ALL keyframes of ALL videos (full scan mode).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- Source paths ---
    p.add_argument(
        "--keyframes_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes",
        help="Dir containing *_keyframes.csv — auto-discovers ALL video IDs from here.",
    )
    p.add_argument(
        "--asr_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr",
        help="Output dir for *_asr.jsonl files (overwritten in place).",
    )
    p.add_argument(
        "--video_dirs",
        type=str,
        nargs="+",
        default=["/mlcv1/Datasets/HCMAI25/full", "/mlcv2025/Datasets/HCMAI25/batch2/video"],
        help="Search paths for source video files (.mp4/.mkv/.avi).",
    )
    p.add_argument(
        "--backup_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr_backup_before_redo",
        help="Backup old *_asr.jsonl here before overwriting. Set empty to disable.",
    )

    # --- Filtering / resume ---
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip videos that already have a *_asr.jsonl (useful to resume partial runs).",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Skip all videos alphabetically BEFORE this video_id (e.g. --resume_from L24_V001).",
    )

    # --- GPU / model ---
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value (e.g. 0, 1, 0,1).")
    p.add_argument(
        "--asr_model_size",
        type=str,
        default="large-v3-turbo",
        choices=[
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en", "distil-small.en",
            "medium", "medium.en", "distil-medium.en",
            "large-v1", "large-v2", "large-v3", "large",
            "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo",
        ],
        help=(
            "faster-whisper model size. Default: large-v3-turbo (~6GB VRAM float16).\n"
            "  large-v3-turbo : ~6GB float16, near-same WER as large-v3 (RECOMMENDED)\n"
            "  large-v2       : ~8-9GB float16, highest quality\n"
            "  large-v3       : ~8-9GB float16, same as large-v2 but newer\n"
            "  medium         : ~5GB float16"
        ),
    )
    p.add_argument(
        "--asr_compute_type",
        type=str,
        default="float16",
        choices=["float16", "float32", "int8", "int8_float16", "int16", "bfloat16"],
        help=(
            "faster-whisper compute type.\n"
            "  float16      : Best speed+quality on GPU (RECOMMENDED for <10GB VRAM)\n"
            "  int8_float16 : ~Half VRAM vs float16, slightly lower quality\n"
            "  int8         : CPU-friendly, lowest VRAM"
        ),
    )
    p.add_argument("--asr_language", type=str, default="vi", help="Whisper language code.")
    p.add_argument("--window_sec", type=float, default=3.0, help="Half-window (sec) around keyframe timestamp for ASR mapping.")
    p.add_argument("--beam_size", type=int, default=5, help="Whisper beam size (5 = default, higher = slower + slightly better).")
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="faster-whisper num_workers (parallel audio decoding). Keep 1 for single-GPU.",
    )

    # --- Misc ---
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan only — no transcription. Use to verify paths before running.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def discover_all_video_ids(keyframes_metadata_dir: str):
    """Auto-discover ALL video IDs from *_keyframes.csv files."""
    pattern = os.path.join(keyframes_metadata_dir, "*_keyframes.csv")
    files = sorted(glob.glob(pattern))
    video_ids = []
    for f in files:
        basename = os.path.basename(f)
        # Strip _keyframes.csv suffix
        vid = basename.replace("_keyframes.csv", "")
        if vid:
            video_ids.append(vid)
    return video_ids


def find_video_path(video_id, video_dirs):
    for ext in [".mp4", ".mkv", ".avi"]:
        for d in video_dirs:
            p = os.path.join(d, f"{video_id}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_all_keyframes(video_id, keyframes_metadata_dir):
    """Load ALL keyframes for this video from CSV (and JSON fallback)."""
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
    if not backup_dir:
        return None
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

    # --- 1. Discover ALL video IDs from keyframes metadata dir ---
    all_video_ids = discover_all_video_ids(args.keyframes_metadata_dir)
    if not all_video_ids:
        print(f"[-] No *_keyframes.csv found in: {args.keyframes_metadata_dir}")
        sys.exit(2)

    print(f"[SCAN] Found {len(all_video_ids)} video IDs in {args.keyframes_metadata_dir}")

    # --- 2. Apply --resume_from filter ---
    if args.resume_from:
        before = len(all_video_ids)
        all_video_ids = [v for v in all_video_ids if v >= args.resume_from]
        print(f"[RESUME] Skipping {before - len(all_video_ids)} videos before '{args.resume_from}' -> {len(all_video_ids)} remaining")

    # --- 3. Apply --skip_existing filter ---
    if args.skip_existing:
        before = len(all_video_ids)
        all_video_ids = [
            v for v in all_video_ids
            if not os.path.exists(os.path.join(args.asr_metadata_dir, f"{v}_asr.jsonl"))
        ]
        print(f"[SKIP_EXISTING] Skipped {before - len(all_video_ids)} already-done -> {len(all_video_ids)} remaining")

    if not all_video_ids:
        print("[+] Nothing to do. All videos already processed.")
        return

    # --- 4. Build plan ---
    print(f"\n[PLAN] Processing {len(all_video_ids)} videos")
    print(f"       GPU={args.gpu}   model={args.asr_model_size}   compute={args.asr_compute_type}")
    print(f"       lang={args.asr_language}   beam={args.beam_size}   window=±{args.window_sec}s")
    print(f"       keyframes_dir : {args.keyframes_metadata_dir}")
    print(f"       asr_out_dir   : {args.asr_metadata_dir}")
    print(f"       backup_dir    : {args.backup_dir or '(disabled)'}")
    print(f"       video_dirs    : {args.video_dirs}")

    plan = []
    for vid in all_video_ids:
        vp = find_video_path(vid, args.video_dirs)
        kfs = load_all_keyframes(vid, args.keyframes_metadata_dir)
        old_asr_exists = os.path.exists(os.path.join(args.asr_metadata_dir, f"{vid}_asr.jsonl"))
        plan.append({
            "video_id": vid,
            "video_path": vp,
            "num_kf": len(kfs),
            "old_asr_exists": old_asr_exists,
        })

    missing_video = [x for x in plan if not x["video_path"]]
    no_kf = [x for x in plan if x["num_kf"] == 0]
    ok_plan = [x for x in plan if x["video_path"] and x["num_kf"] > 0]
    total_kf = sum(x["num_kf"] for x in ok_plan)

    print(f"\n[DRY_SCAN] Total={len(plan)} | OK={len(ok_plan)} | Missing_video={len(missing_video)} | No_kf={len(no_kf)}")
    print(f"           Total keyframes to process: {total_kf:,}")
    if missing_video:
        print(f"  [!] Videos with no video file found ({len(missing_video)}):")
        for x in missing_video[:10]:
            print(f"      {x['video_id']}")
        if len(missing_video) > 10:
            print(f"      ... and {len(missing_video) - 10} more")
    if no_kf:
        print(f"  [!] Videos with 0 keyframes ({len(no_kf)}):")
        for x in no_kf[:10]:
            print(f"      {x['video_id']}")

    if args.dry_run:
        print("\n[DRY_RUN] DONE - no transcription ran. Remove --dry_run to execute.")
        return

    # --- 5. Load Whisper model ---
    import torch
    from faster_whisper import WhisperModel

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    compute_type = args.asr_compute_type
    if not cuda_available and compute_type == "float16":
        print("[WARN] CUDA not available, falling back compute_type=int8")
        compute_type = "int8"

    print(f"\n[INIT] Loading WhisperModel '{args.asr_model_size}' | device={device} | compute={compute_type} | num_workers={args.num_workers}")
    print(f"       torch={torch.__version__}  cuda={cuda_available}")
    model = WhisperModel(
        args.asr_model_size,
        device=device,
        compute_type=compute_type,
        num_workers=args.num_workers,
    )
    print("[INIT] Model loaded.\n")

    # --- 6. Process ---
    run_log = []
    t_start_all = time.perf_counter()
    processed_ok = 0

    for idx, item in enumerate(ok_plan, 1):
        vid = item["video_id"]
        prefix = f"[{idx}/{len(ok_plan)}] {vid}"
        print(f"\n{prefix}  kf={item['num_kf']}  size={os.path.getsize(item['video_path'])/1e9:.2f}GB  {os.path.basename(item['video_path'])}")

        # Backup old asr
        bkp = backup_old_file(vid, args.asr_metadata_dir, args.backup_dir)
        if bkp:
            print(f"  backup -> {bkp}")

        # Transcribe
        t0 = time.perf_counter()
        try:
            segments, _ = model.transcribe(
                item["video_path"],
                language=args.asr_language,
                beam_size=args.beam_size,
                condition_on_previous_text=False,
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
            print(f"  whisper: {len(raw_segments)} segs (w/ text={n_speech}) in {t_trans:.1f}s")
        except Exception as e:
            print(f"  [ERROR] transcribe failed: {e}")
            run_log.append({"video_id": vid, "status": "error", "error": str(e)})
            continue

        # Map keyframes -> ASR window
        kf_all = load_all_keyframes(vid, args.keyframes_metadata_dir)
        records = [map_asr_to_keyframe(row, raw_segments, args.window_sec) for row in kf_all.values()]
        n_has_speech = sum(1 for r in records if r["asr_metadata"].get("has_speech"))
        unique_non_empty = {r["asr_metadata"].get("asr_text_6s", "") for r in records} - {""}
        print(f"  mapped: {len(records)} recs | has_speech={n_has_speech} | unique_texts={len(unique_non_empty)}")

        # Save
        out_path, n_written = save_asr_records(vid, records, args.asr_metadata_dir)
        dur = time.perf_counter() - t0
        processed_ok += 1
        eta_per_vid = dur
        remaining = len(ok_plan) - idx
        print(f"  SAVED -> {out_path} ({n_written} recs) | total={dur:.1f}s | ETA ~{remaining * eta_per_vid / 60:.0f}min")

        run_log.append({
            "video_id": vid,
            "status": "ok",
            "kf_count": n_written,
            "has_speech": n_has_speech,
            "unique_non_empty_asr": len(unique_non_empty),
            "whisper_segments": len(raw_segments),
            "elapsed_sec": round(dur, 2),
            "backup": bkp,
            "output": out_path,
        })

    # --- 7. Summary ---
    total_elapsed = time.perf_counter() - t_start_all
    status_counts = {}
    for x in run_log:
        s = x["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print("\n" + "=" * 90)
    print(f"[ALL DONE] processed={processed_ok}/{len(ok_plan)} videos in {total_elapsed/60:.1f}min ({total_elapsed:.0f}s)")
    print(f"           skipped (no video/kf): {len(missing_video) + len(no_kf)}")
    print(f"           status: {status_counts}")
    print(f"           model={args.asr_model_size}  compute={compute_type}  device={device}")
    print("=" * 90)

    # Save JSON log
    log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"redo_asr_log_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {k: v for k, v in vars(args).items()},
                "summary": {
                    "n_total": len(all_video_ids),
                    "n_ok": processed_ok,
                    "n_error": status_counts.get("error", 0),
                    "n_skipped_no_video": len(missing_video),
                    "n_skipped_no_kf": len(no_kf),
                    "total_kf": total_kf,
                    "elapsed_sec": round(total_elapsed, 2),
                    "device": device,
                    "compute_type": compute_type,
                    "model_size": args.asr_model_size,
                },
                "per_video": run_log,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
