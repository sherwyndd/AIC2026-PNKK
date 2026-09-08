"""Verify Qdrant payload image_path actually exist on disk.

Usage:
  conda activate aic2026_backend
  cd UI/BackEnd
  python tests/_verify_qdrant_image_paths.py [--limit 50000] [--collection image_siglip]

Output:
  - Console summary: total checked / exists / missing + % integrity
  - tests/_broken_image_paths.txt: list of <keyframe_id>\t<image_path> rows (missing only)
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from tqdm import tqdm


KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BATCH_SIZE = 1000
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_broken_image_paths.txt")


def parse_args():
    p = argparse.ArgumentParser(description="Verify Qdrant payload image_path on disk")
    p.add_argument("--collection", default="image_siglip", help="Qdrant collection name")
    p.add_argument("--limit", type=int, default=0, help="Max points to check (0 = all)")
    p.add_argument("--batch", type=int, default=BATCH_SIZE, help="Scroll batch size")
    return p.parse_args()


def main():
    args = parse_args()
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    info = client.get_collection(args.collection)
    total = info.points_count
    print(f"Collection: {args.collection}")
    print(f"Points count reported: {total}")
    print(f"Keyframe root: {KEYFRAME_ROOT}")
    print(f"Keyframe root exists: {os.path.isdir(KEYFRAME_ROOT)}")

    limit = args.limit if args.limit > 0 else total
    checked = 0
    exists = 0
    missing = 0
    missing_rows = []
    missing_per_video = Counter()

    offset = None
    pbar = tqdm(total=min(total, limit), desc="Scanning Qdrant", unit="p")
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=args.collection,
                limit=min(args.batch, limit - checked),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            print(f"\nScroll failed at offset={offset}: {e}")
            break

        if not records:
            break

        for rec in records:
            checked += 1
            p = rec.payload or {}
            image_path = p.get("image_path") or ""
            full = os.path.join(KEYFRAME_ROOT, image_path) if image_path else ""
            if image_path and os.path.isfile(full):
                exists += 1
            else:
                missing += 1
                vid = p.get("video_id") or ""
                kid = p.get("keyframe_id") or rec.id
                missing_rows.append(f"{kid}\t{vid}\t{image_path}")
                if vid:
                    missing_per_video[vid] += 1

            pbar.update(1)
            if checked >= limit:
                break

        offset = next_offset
        if offset is None or checked >= limit:
            break
    pbar.close()

    print("\n" + "=" * 60)
    print(f"Checked      : {checked}")
    print(f"Exists       : {exists}  ({(exists/checked*100) if checked else 0:.2f}%)")
    print(f"Missing      : {missing}  ({(missing/checked*100) if checked else 0:.2f}%)")

    if missing_rows:
        with open(OUT_FILE, "w") as f:
            f.write("keyframe_id\tvideo_id\timage_path\n")
            for r in missing_rows:
                f.write(r + "\n")
        print(f"\nMissing list written to: {OUT_FILE}")
        print(f"Top 10 videos with most missing frames:")
        for vid, cnt in missing_per_video.most_common(10):
            print(f"  {vid or '(unknown)':<20} -> {cnt} missing")
    else:
        print("\nAll image_path files found on disk. Integrity OK.")
        if os.path.exists(OUT_FILE):
            os.remove(OUT_FILE)


if __name__ == "__main__":
    main()
