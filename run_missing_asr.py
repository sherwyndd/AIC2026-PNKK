#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# CUDA library and Environment variable resolution
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
# Setup CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and update ASR metadata specifically for missing keyframes."
    )
    parser.add_argument(
        "--missing_path",
        type=str,
        default="/AIClub_NAS/core_baotg/phong/AIC_2026/UI/missing_keyframes.txt",
        help="Path to the file listing missing keyframe IDs."
    )
    parser.add_argument(
        "--video_dirs",
        type=str,
        nargs="+",
        default=["/mlcv1/Datasets/HCMAI25/full", "/mlcv2025/Datasets/HCMAI25/batch2/video"],
        help="Directories containing video files."
    )
    parser.add_argument(
        "--keyframes_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes",
        help="Directory containing the keyframe metadata CSV/JSON files."
    )
    parser.add_argument(
        "--asr_metadata_dir",
        type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr",
        help="Directory containing the ASR metadata JSON/JSONL files."
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (e.g. 0)"
    )
    parser.add_argument(
        "--asr_model_size",
        type=str,
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="faster-whisper model size to use."
    )
    parser.add_argument(
        "--asr_language",
        type=str,
        default="vi",
        help="Language code for ASR."
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=3.0,
        help="Half-window size in seconds to map speech to keyframe timestamp."
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for transcription."
    )
    return parser.parse_args()


def load_missing_keyframes(path):
    if not os.path.exists(path):
        print(f"[-] Error: Missing keyframes file not found: {path}")
        sys.exit(1)
        
    missing_by_video = {}
    total_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kf_id = line.strip()
            if not kf_id or kf_id.startswith("#"):
                continue
            if "_kf_" in kf_id:
                video_id = kf_id.split("_kf_")[0]
                missing_by_video.setdefault(video_id, []).append(kf_id)
                total_count += 1
            else:
                print(f"[!] Warning: Invalid keyframe ID format: {kf_id}")
    print(f"[+] Loaded {total_count} missing keyframes across {len(missing_by_video)} videos from {path}.")
    return missing_by_video


def find_video_path(video_id, video_dirs):
    extensions = [".mp4", ".mkv", ".avi"]
    for d in video_dirs:
        for ext in extensions:
            path = os.path.join(d, f"{video_id}{ext}")
            if os.path.exists(path):
                return path
    return None


def load_keyframes_data(video_id, target_kf_ids, keyframes_metadata_dir):
    csv_path = os.path.join(keyframes_metadata_dir, f"{video_id}_keyframes.csv")
    json_path = os.path.join(keyframes_metadata_dir, f"{video_id}_keyframes.json")
    
    data = {}
    
    # Try CSV
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kf_id = row.get("keyframe_id", "").strip()
                if kf_id in target_kf_ids:
                    # Parse integers and floats
                    try:
                        row["frame_idx"] = int(row["frame_idx"])
                    except (ValueError, TypeError, KeyError):
                        row["frame_idx"] = 0
                    try:
                        row["pts_time"] = float(row["pts_time"])
                    except (ValueError, TypeError, KeyError):
                        row["pts_time"] = 0.0
                    try:
                        row["timestamp"] = int(row["timestamp"])
                    except (ValueError, TypeError, KeyError):
                        row["timestamp"] = int(round(row["pts_time"] * 1000))
                    data[kf_id] = row
                    
    # Try JSON if CSV didn't yield all target keyframes or doesn't exist
    if len(data) < len(target_kf_ids) and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
                for row in records:
                    kf_id = row.get("keyframe_id", "").strip()
                    if kf_id in target_kf_ids and kf_id not in data:
                        try:
                            row["frame_idx"] = int(row["frame_idx"])
                        except (ValueError, TypeError, KeyError):
                            row["frame_idx"] = 0
                        try:
                            row["pts_time"] = float(row["pts_time"])
                        except (ValueError, TypeError, KeyError):
                            row["pts_time"] = 0.0
                        try:
                            row["timestamp"] = int(row["timestamp"])
                        except (ValueError, TypeError, KeyError):
                            row["timestamp"] = int(round(row["pts_time"] * 1000))
                        data[kf_id] = row
            except Exception as e:
                print(f"[!] Error reading JSON keyframes for {video_id}: {e}")
                
    return data


def load_existing_asr(video_id, asr_metadata_dir):
    jsonl_path = os.path.join(asr_metadata_dir, f"{video_id}_asr.jsonl")
    json_path = os.path.join(asr_metadata_dir, f"{video_id}_asr.json")
    
    records = []
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records, "jsonl"
        
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                pass
        return records, "json"
        
    return [], "jsonl"


def save_asr_records(video_id, records, fmt, asr_metadata_dir):
    os.makedirs(asr_metadata_dir, exist_ok=True)
    jsonl_path = os.path.join(asr_metadata_dir, f"{video_id}_asr.jsonl")
    json_path = os.path.join(asr_metadata_dir, f"{video_id}_asr.json")
    
    # Sort records by frame_idx for consistency
    def get_frame_idx(rec):
        try:
            return int(rec.get("frame_idx", 0))
        except (ValueError, TypeError):
            return 0
    records = sorted(records, key=get_frame_idx)
    
    if fmt == "jsonl":
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[+] Updated {len(records)} ASR records in JSONL: {jsonl_path}")
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        print(f"[+] Updated {len(records)} ASR records in JSON: {json_path}")


def map_asr_to_keyframe(row, raw_segments, window_sec):
    try:
        t_kf = float(row.get("pts_time", 0.0))
    except (ValueError, TypeError):
        t_kf = 0.0
        
    w_start = max(0.0, t_kf - window_sec)
    w_end = t_kf + window_sec
    matches = []
    for seg in raw_segments:
        try:
            seg_start = float(seg["segment_start"])
            seg_end = float(seg["segment_end"])
        except (ValueError, TypeError, KeyError):
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
        }
    }


def main():
    args = parse_args()
    
    # Force GPU configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Load missing keyframes list
    missing_by_video = load_missing_keyframes(args.missing_path)
    if not missing_by_video:
        print("[+] No missing keyframes to process. Exiting.")
        return
        
    # Import libraries inside main so they don't load before LD_LIBRARY_PATH is set
    import torch
    from faster_whisper import WhisperModel
    
    print(f"[+] Initializing WhisperModel size={args.asr_model_size} on GPU={args.gpu}...")
    cuda_available = torch.cuda.is_available()
    compute_type = "float16" if cuda_available else "int8"
    device = "cuda" if cuda_available else "cpu"
    print(f"    PyTorch CUDA available: {cuda_available} (Running on {device})")
    
    model = WhisperModel(
        args.asr_model_size,
        device=device,
        compute_type=compute_type,
        num_workers=1
    )
    print("[+] WhisperModel loaded successfully.")
    
    processed_count = 0
    
    for idx, (video_id, kf_ids) in enumerate(missing_by_video.items()):
        print(f"\n[{idx+1}/{len(missing_by_video)}] Processing video: {video_id} ({len(kf_ids)} missing keyframes)")
        
        # 1. Find video path
        video_path = find_video_path(video_id, args.video_dirs)
        if not video_path:
            print(f"    [!] Warning: Video file for {video_id} not found in {args.video_dirs}. Skipping.")
            continue
        print(f"    Found video: {video_path}")
        
        # 2. Load keyframes metadata
        keyframes_info = load_keyframes_data(video_id, kf_ids, args.keyframes_metadata_dir)
        if not keyframes_info:
            print(f"    [!] Warning: No keyframe metadata found for {video_id} in {args.keyframes_metadata_dir}. Skipping.")
            continue
        print(f"    Loaded info for {len(keyframes_info)}/{len(kf_ids)} missing keyframes.")
        
        # 3. Transcribe video
        print(f"    Transcribing video with Whisper...")
        t_trans_start = time.perf_counter()
        try:
            segments, _ = model.transcribe(video_path, language=args.asr_language, beam_size=args.beam_size)
            raw_segments = [
                {
                    "segment_start": float(s.start),
                    "segment_end": float(s.end),
                    "text": s.text.strip(),
                }
                for s in segments
            ]
            elapsed_trans = time.perf_counter() - t_trans_start
            print(f"    Transcription complete in {elapsed_trans:.2f}s. Extracted {len(raw_segments)} speech segments.")
        except Exception as e:
            print(f"    [!] Error transcribing {video_id}: {e}. Skipping.")
            continue
            
        # 4. Map to target keyframes
        mapped_records = {}
        for kf_id, kf_row in keyframes_info.items():
            mapped_records[kf_id] = map_asr_to_keyframe(kf_row, raw_segments, args.window_sec)
            
        # 5. Load existing ASR records
        existing_records, fmt = load_existing_asr(video_id, args.asr_metadata_dir)
        print(f"    Found {len(existing_records)} existing records (Format: {fmt}).")
        
        # 6. Merge & update records
        merged_records_dict = {}
        for rec in existing_records:
            k_id = rec.get("keyframe_id")
            if k_id:
                merged_records_dict[k_id] = rec
                
        # Overwrite with newly generated missing records
        for k_id, new_rec in mapped_records.items():
            merged_records_dict[k_id] = new_rec
            
        merged_records = list(merged_records_dict.values())
        
        # 7. Save back to output directory
        save_asr_records(video_id, merged_records, fmt, args.asr_metadata_dir)
        processed_count += len(mapped_records)
        
    print(f"\n[+] Finished. Successfully generated ASR for {processed_count} keyframes.")


if __name__ == "__main__":
    main()
