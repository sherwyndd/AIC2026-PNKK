import os
import glob
import json
import time
from collections import defaultdict

def _sort_key(rec):
    try:
        fi = rec.get("frame_idx", 0)
        fi = int(fi) if fi is not None else 0
    except (TypeError, ValueError):
        fi = 0
    return (fi, rec.get("keyframe_id", ""))

def main():
    asr_dir = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"

    pattern_jsonl = os.path.join(asr_dir, "*.jsonl")
    pattern_json = os.path.join(asr_dir, "*.json")

    files = glob.glob(pattern_jsonl) + glob.glob(pattern_json)
    files.sort()

    total_files = len(files)
    print(f"Found {total_files} files to process.")

    video_records = defaultdict(list)
    start_time = time.time()
    total_read = 0

    for idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)

        if idx == 1 or idx % 50 == 0 or idx == total_files:
            elapsed = time.time() - start_time
            print(f"[READ {idx}/{total_files}] {filename} read={total_read} elapsed={elapsed:.1f}s")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                if filepath.endswith(".jsonl"):
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError as je:
                            print(f"  Skip bad json {filename}:{line_num}: {je}")
                            continue
                        kf_id = rec.get("keyframe_id")
                        video_id = rec.get("video_id")
                        if not kf_id:
                            continue
                        if not video_id:
                            parts = kf_id.rsplit("_kf_", 1)
                            if len(parts) == 2:
                                video_id = parts[0]
                                rec["video_id"] = video_id
                            else:
                                continue
                        video_records[video_id].append(rec)
                        total_read += 1
                else:
                    try:
                        content = json.load(f)
                    except json.JSONDecodeError as je:
                        print(f"  Skip bad json {filename}: {je}")
                        continue
                    items = content if isinstance(content, list) else [content]
                    for rec in items:
                        kf_id = rec.get("keyframe_id")
                        video_id = rec.get("video_id")
                        if not kf_id:
                            continue
                        if not video_id:
                            parts = kf_id.rsplit("_kf_", 1)
                            if len(parts) == 2:
                                video_id = parts[0]
                                rec["video_id"] = video_id
                            else:
                                continue
                        video_records[video_id].append(rec)
                        total_read += 1
        except Exception as e:
            print(f"  Read error {filename}: {e}")

    print(f"\nRead done: {total_read} keyframes across {len(video_records)} videos.")

    total_written_files = 0
    total_written_kf = 0
    start_w = time.time()
    video_ids = sorted(video_records.keys())
    total_videos = len(video_ids)

    for i, vid in enumerate(video_ids, 1):
        recs = video_records[vid]
        recs.sort(key=_sort_key)

        out_path = os.path.join(asr_dir, f"{vid}_asr.jsonl")

        try:
            with open(out_path, "w", encoding="utf-8") as fo:
                for rec in recs:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_written_files += 1
            total_written_kf += len(recs)
        except Exception as e:
            print(f"  Write error {out_path}: {e}")

        if i == 1 or i % 50 == 0 or i == total_videos:
            elapsed = time.time() - start_w
            print(f"[WRITE {i}/{total_videos}] {vid}_asr.jsonl ({len(recs)} kf) elapsed={elapsed:.1f}s")

    total_videos_out = len(video_records)
    total_all = total_written_kf
    print(f"\n=== Done ===")
    print(f"Videos rewritten: {total_written_files}/{total_videos_out}")
    print(f"Total keyframes written (to *_asr.jsonl): {total_written_kf}")
    print(f"Output dir: {asr_dir}")

if __name__ == "__main__":
    main()
