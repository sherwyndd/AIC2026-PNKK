import os
import csv
import json
import glob
import time
from collections import defaultdict

VIDEOS_CSV = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv"
MEDIA_INFO_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/media-info"
ASR_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"
OUTPUT_CSV = "/AIClub_NAS/core_baotg/phong/AIC_2026/UI/videos_metadata_filtered.csv"

def load_videos_csv():
    videos = {}
    with open(VIDEOS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            videos[vid] = row
    return videos

def load_media_info(video_ids):
    media = {}
    for vid in video_ids:
        path = os.path.join(MEDIA_INFO_DIR, f"{vid}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                media[vid] = {
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "author": data.get("author", ""),
                    "publish_date": data.get("publish_date", ""),
                    "keywords": "|".join(data.get("keywords", []) or []),
                    "watch_url": data.get("watch_url", ""),
                }
            except Exception:
                pass
        if vid not in media:
            media[vid] = {
                "title": "", "description": "", "author": "",
                "publish_date": "", "keywords": "", "watch_url": "",
            }
    return media

def load_asr_stats(video_ids):
    stats = defaultdict(lambda: {
        "total_keyframes": 0,
        "has_asr": 0,
        "no_asr": 0,
        "has_speech": 0,
        "no_speech": 0,
    })
    pattern_jsonl = os.path.join(ASR_DIR, "*_asr.jsonl")
    pattern_json = os.path.join(ASR_DIR, "*_asr.json")
    files = glob.glob(pattern_jsonl) + glob.glob(pattern_json)
    files.sort()
    total = len(files)
    start = time.time()
    for idx, fp in enumerate(files, 1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[ASR {idx}/{total}] {os.path.basename(fp)} elapsed={time.time()-start:.1f}s")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                if fp.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        vid = d.get("video_id")
                        if not vid or vid not in video_ids:
                            continue
                        s = stats[vid]
                        s["total_keyframes"] += 1
                        asr_meta = d.get("asr_metadata") or {}
                        text = asr_meta.get("asr_text_6s")
                        has_text = bool(text and isinstance(text, str) and text.strip())
                        if has_text:
                            s["has_asr"] += 1
                        else:
                            s["no_asr"] += 1
                        if asr_meta.get("has_speech") is True:
                            s["has_speech"] += 1
                        else:
                            s["no_speech"] += 1
                else:
                    try:
                        content = json.load(f)
                    except Exception:
                        continue
                    items = content if isinstance(content, list) else [content]
                    for d in items:
                        vid = d.get("video_id")
                        if not vid or vid not in video_ids:
                            continue
                        s = stats[vid]
                        s["total_keyframes"] += 1
                        asr_meta = d.get("asr_metadata") or {}
                        text = asr_meta.get("asr_text_6s")
                        has_text = bool(text and isinstance(text, str) and text.strip())
                        if has_text:
                            s["has_asr"] += 1
                        else:
                            s["no_asr"] += 1
                        if asr_meta.get("has_speech") is True:
                            s["has_speech"] += 1
                        else:
                            s["no_speech"] += 1
        except Exception:
            continue
    for vid in video_ids:
        if vid not in stats:
            stats[vid] = {
                "total_keyframes": 0, "has_asr": 0, "no_asr": 0,
                "has_speech": 0, "no_speech": 0,
            }
    return stats

def main():
    print("Loading videos_metadata.csv ...")
    videos = load_videos_csv()
    print(f"  -> {len(videos)} videos")

    video_ids = list(videos.keys())

    print("Loading media-info JSONs ...")
    media = load_media_info(video_ids)
    print(f"  -> {len([v for v in media.values() if v['title']])} has title")

    print("Aggregating ASR stats ...")
    stats = load_asr_stats(video_ids)

    fields = [
        "video_id", "file_name", "fps", "total_frames", "duration",
        "width", "height", "bitrate_kbps", "backend", "format", "fourcc",
        "title", "author", "publish_date", "keywords", "watch_url",
        "total_keyframes", "has_asr", "no_asr", "has_speech", "no_speech",
        "description",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for vid in video_ids:
            v = videos[vid]
            m = media.get(vid, {})
            s = stats.get(vid, {})
            row = {
                "video_id": v.get("video_id", ""),
                "file_name": v.get("file_name", ""),
                "fps": v.get("fps", ""),
                "total_frames": v.get("total_frames", ""),
                "duration": v.get("duration", ""),
                "width": v.get("width", ""),
                "height": v.get("height", ""),
                "bitrate_kbps": v.get("bitrate_kbps", ""),
                "backend": v.get("backend", ""),
                "format": v.get("format", ""),
                "fourcc": v.get("fourcc", ""),
                "title": m.get("title", ""),
                "author": m.get("author", ""),
                "publish_date": m.get("publish_date", ""),
                "keywords": m.get("keywords", ""),
                "watch_url": m.get("watch_url", ""),
                "total_keyframes": s.get("total_keyframes", 0),
                "has_asr": s.get("has_asr", 0),
                "no_asr": s.get("no_asr", 0),
                "has_speech": s.get("has_speech", 0),
                "no_speech": s.get("no_speech", 0),
                "description": m.get("description", ""),
            }
            writer.writerow(row)

    total_v = len(video_ids)
    total_kf = sum(stats[v]["total_keyframes"] for v in video_ids)
    total_has = sum(stats[v]["has_asr"] for v in video_ids)
    total_no = sum(stats[v]["no_asr"] for v in video_ids)
    print()
    print(f"Tong cong: {total_v} videos, {total_kf} keyframes")
    print(f"  Co ASR:   {total_has} ({(total_has/total_kf*100 if total_kf else 0):.1f}%)")
    print(f"  Ko ASR:   {total_no} ({(total_no/total_kf*100 if total_kf else 0):.1f}%)")
    print(f"\nDa ghi ra: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
