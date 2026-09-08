"""Fast importer: load pre-computed SigLIP2 .npy embeddings (from nhan's pipeline) into Qdrant.

Processes each video folder ONE AT A TIME (no full concat → low memory, instant feedback).
Shows tqdm progress bars for both folders and per-folder insertions.

Usage:
    cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd
    python fast_import_npy_to_qdrant.py
"""

import os
import sys
import re
import hashlib
import logging
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Paths (absolute so it works from any cwd) ────────────────────────────────
EMBEDDING_DIR = Path("/AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/embeddings")
KEYFRAME_ROOT = Path("/AIClub_NAS/core_baotg/nhan/dataset/keyframes")
COLLECTION_NAME = "image_siglip"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BATCH_SIZE = 500  # Upsert batch size per Qdrant call

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fast_import")


def parse_image_meta(image_path: str):
    stem = Path(image_path).stem
    m = re.match(r"(L\d+_V\d+)_kf_(\d+)", stem)
    if m is None:
        log.warning("Cannot parse filename (fallback): %s", stem)
        return stem, 0
    return m.group(1), int(m.group(2))


def ensure_collection(client: QdrantClient, recreate: bool = False):
    """Create collection if missing, optionally recreate from scratch."""
    cols = [c.name for c in client.get_collections().collections]
    exists = COLLECTION_NAME in cols
    if exists and recreate:
        log.info("Deleting old collection '%s' ...", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        log.info("Creating collection '%s' (dim=768, distance=COSINE) ...", COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    else:
        log.info("Collection '%s' already exists, will resume (skip existing by id).", COLLECTION_NAME)


def list_video_dirs() -> List[Path]:
    dirs = sorted(
        [d for d in EMBEDDING_DIR.iterdir()
         if d.is_dir() and (d / "image_embeddings.npy").exists() and (d / "image_paths.txt").exists()]
    )
    log.info("Found %d video folders with embeddings in %s", len(dirs), EMBEDDING_DIR)
    return dirs


def count_existing_points(client: QdrantClient) -> int:
    try:
        return client.count(collection_name=COLLECTION_NAME, exact=True).count
    except Exception:
        return 0


def main():
    log.info("Connecting to Qdrant %s:%s ...", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

    # If collection already has points → recreate to be safe (vectors are guaranteed correct from .npy)
    current = count_existing_points(client)
    recreate = current < 1000  # if too few, start clean; else keep as-is with resume safety
    ensure_collection(client, recreate=recreate)

    dirs = list_video_dirs()
    if not dirs:
        log.error("No embedding folders found. Exit.")
        sys.exit(1)

    total_vectors_expected = 0
    for d in dirs:
        with open(d / "image_paths.txt") as f:
            total_vectors_expected += sum(1 for line in f if line.strip())
    log.info("Total vectors expected from all folders: %d", total_vectors_expected)

    pbar = tqdm(dirs, desc="Overall (video folders)", unit="folder")
    total_upserted = 0
    total_skipped = 0

    for folder in pbar:
        pbar.set_postfix_str(folder.name)
        npy_file = folder / "image_embeddings.npy"
        txt_file = folder / "image_paths.txt"

        # Load this folder's data (~1MB per folder, instant)
        try:
            embeddings = np.load(npy_file).astype(np.float32)  # (N, 768)
        except Exception as e:
            log.error("Skip %s: bad npy: %s", folder.name, e)
            continue

        with open(txt_file) as f:
            paths_raw = [line.strip() for line in f if line.strip()]

        N = embeddings.shape[0]
        if len(paths_raw) != N:
            log.warning("  %s: paths=%d != embs=%d → truncating", folder.name, len(paths_raw), N)
            paths_raw = paths_raw[:N]
            N = min(len(paths_raw), N)
            embeddings = embeddings[:N]

        # Build and upsert in batches
        batch_points = []
        folder_upserted = 0

        for idx in range(N):
            abs_path = Path(paths_raw[idx])
            try:
                rel_path = str(abs_path.relative_to(KEYFRAME_ROOT))
            except ValueError:
                rel_path = str(abs_path)

            video_id, frame_idx = parse_image_meta(str(abs_path))
            doc_id = int.from_bytes(
                hashlib.md5(rel_path.encode()).digest()[:8], "big",
            )

            batch_points.append(PointStruct(
                id=doc_id,
                vector=embeddings[idx].tolist(),
                payload={
                    "video_id": video_id,
                    "keyframe_id": Path(rel_path).stem,
                    "frame_idx": frame_idx,
                    "image_path": rel_path,
                },
            ))

            if len(batch_points) >= BATCH_SIZE:
                try:
                    client.upsert(collection_name=COLLECTION_NAME, points=batch_points, wait=True)
                except Exception as e:
                    log.error("  Upsert batch failed in %s: %s", folder.name, e)
                folder_upserted += len(batch_points)
                total_upserted += len(batch_points)
                batch_points = []

        # Flush remainder
        if batch_points:
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=batch_points, wait=True)
                folder_upserted += len(batch_points)
                total_upserted += len(batch_points)
            except Exception as e:
                log.error("  Upsert final batch failed in %s: %s", folder.name, e)

        total_skipped += (N - folder_upserted)
        pbar.set_postfix_str(f"{folder.name} | N={N} up={total_upserted}")

    pbar.close()

    final_count = count_existing_points(client)
    log.info("=" * 60)
    log.info("DONE. Collection '%s' has %d points.", COLLECTION_NAME, final_count)
    log.info("  Reported upserts: %d  Skipped: %d", total_upserted, total_skipped)
    info = client.get_collection(COLLECTION_NAME)
    log.info("  Distance: %s  |  Dim: %s",
             info.config.params.vectors.distance,
             info.config.params.vectors.size)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
