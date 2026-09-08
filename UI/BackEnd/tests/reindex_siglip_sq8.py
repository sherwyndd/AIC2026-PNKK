"""Re-create SigLIP collection WITH Scalar Quantization 8-bit + ef_construct=256.

REUSES pre-computed .npy embeddings from nhan's pipeline (no re-encoding! fast).
Creates a NEW collection 'image_siglip_sq8' so we can test before swapping production.

Config applied:
  - Vectors: dim=768, distance=COSINE, on_disk=false (RAM only - 488GB free)
  - Quantization: ScalarQuantization INT8 (always_ram=true) -> -75% vector memory
  - HNSW:         m=16 (default), ef_construct=256 (recall +0.5-1.5% to compensate SQ loss)
  - Optimizer:    indexing_threshold=20000 (optimize sooner for fast search)

Usage:
    cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd
    python3 tests/reindex_siglip_sq8.py
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
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,          # Corrected: SQ type = ScalarType.INT8, not QuantizationType
    HnswConfigDiff,
    OptimizersConfigDiff,
)

# ── Paths (reuse existing .npy) ───────────────────────────────────────────────
EMBEDDING_DIR = Path("/AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/embeddings")
KEYFRAME_ROOT = Path("/AIClub_NAS/core_baotg/nhan/dataset/keyframes")
OLD_COLLECTION = "image_siglip"
NEW_COLLECTION = "image_siglip_sq8"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("sq8_import")


def parse_image_meta(image_path: str):
    stem = Path(image_path).stem
    m = re.match(r"(L\d+_V\d+)_kf_(\d+)", stem)
    if m is None:
        log.warning("Cannot parse filename (fallback): %s", stem)
        return stem, 0
    return m.group(1), int(m.group(2))


def create_sq8_collection(client: QdrantClient):
    """Create image_siglip_sq8 with Scalar INT8 quantization + ef_construct=256."""
    cols = [c.name for c in client.get_collections().collections]
    if NEW_COLLECTION in cols:
        log.warning("Collection %s already exists -> deleting first to start clean.", NEW_COLLECTION)
        client.delete_collection(NEW_COLLECTION)

    log.info(
        "Creating collection '%s' | dim=768 COSINE | ScalarQuantization(INT8, always_ram) | "
        "HNSW ef_construct=256 m=16 | on_disk=false",
        NEW_COLLECTION,
    )

    # Configs
    vectors_cfg = VectorParams(
        size=768,
        distance=Distance.COSINE,
        on_disk=False,              # 488GB RAM: keep vectors in RAM for speed
    )
    hnsw_cfg = HnswConfigDiff(
        m=16,
        ef_construct=256,           # [TUNE #5] ef_construct=256 (default ~128) -> recall +0.5-1.5%
        full_scan_threshold=10000,
    )
    opt_cfg = OptimizersConfigDiff(
        indexing_threshold=20000,    # Build index sooner (batch import 500 -> hit this fast)
        memmap_threshold=1000000,
    )
    quant_cfg = ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,      # Fixed: ScalarType.INT8 replaces QuantizationType.INT8
            always_ram=True,           # [TUNE #4] SQ8 always in RAM (quantized 1B/dim, not a lot)
        ),
    )

    try:
        client.create_collection(
            collection_name=NEW_COLLECTION,
            vectors_config=vectors_cfg,
            hnsw_config=hnsw_cfg,
            optimizers_config=opt_cfg,
            quantization_config=quant_cfg,
        )
        log.info("[OK] Collection '%s' created with Scalar INT8 quantization.", NEW_COLLECTION)
    except TypeError:
        # Fallback if older qdrant-client expects slightly different keyword order
        log.info("[FALLBACK] create_collection API signature incompatible, trying alternate (kwarg-only)...")
        try:
            client.create_collection(
                collection_name=NEW_COLLECTION,
                vectors_config=vectors_cfg,
                hnsw_config=hnsw_cfg,
                optimizers_config=opt_cfg,
                quantization_config=quant_cfg,
            )
        except Exception as e2:
            log.exception("[FATAL] Cannot create SQ8 collection: %s", e2)
            raise


def list_video_dirs() -> List[Path]:
    dirs = sorted(
        [d for d in EMBEDDING_DIR.iterdir()
         if d.is_dir() and (d / "image_embeddings.npy").exists() and (d / "image_paths.txt").exists()]
    )
    log.info("Found %d video folders with .npy embeddings in %s", len(dirs), EMBEDDING_DIR)
    if len(dirs) == 0:
        log.error("No embedding folders found. Check EMBEDDING_DIR path!")
        sys.exit(1)
    return dirs


def count_points(client: QdrantClient, col: str) -> int:
    try:
        return client.count(collection_name=col, exact=True).count
    except Exception:
        return 0


def main():
    log.info("Connecting to Qdrant %s:%s ...", QDRANT_HOST, QDRANT_PORT)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    # Quick verify OLD collection exists
    old_count = count_points(client, OLD_COLLECTION)
    log.info("Reference OLD collection '%s' has %d points (just for verification).", OLD_COLLECTION, old_count)

    # 1) Create NEW SQ8 collection (with ef_construct=256)
    create_sq8_collection(client)

    # 2) Import vectors FROM .npy files (fast, no GPU encoding)
    dirs = list_video_dirs()

    total_vectors_expected = 0
    for d in dirs:
        with open(d / "image_paths.txt") as f:
            total_vectors_expected += sum(1 for line in f if line.strip())
    log.info("Total vectors to import: %d", total_vectors_expected)

    pbar = tqdm(dirs, desc="Import .npy -> image_siglip_sq8", unit="folder")
    total_upserted = 0
    total_skipped = 0

    for folder in pbar:
        pbar.set_postfix_str(folder.name)
        npy_file = folder / "image_embeddings.npy"
        txt_file = folder / "image_paths.txt"

        try:
            embeddings = np.load(npy_file).astype(np.float32)  # (N, 768)
        except Exception as e:
            log.error("Skip %s: bad npy: %s", folder.name, e)
            continue

        with open(txt_file) as f:
            paths_raw = [line.strip() for line in f if line.strip()]

        N = embeddings.shape[0]
        if len(paths_raw) != N:
            log.warning("  %s: paths=%d != embs=%d -> truncating", folder.name, len(paths_raw), N)
            paths_raw = paths_raw[:N]
            N = min(len(paths_raw), N)
            embeddings = embeddings[:N]

        batch_points = []
        folder_upserted = 0

        for idx in range(N):
            abs_path = Path(paths_raw[idx])
            try:
                rel_path = str(abs_path.relative_to(KEYFRAME_ROOT))
            except ValueError:
                rel_path = str(abs_path)

            video_id, frame_idx = parse_image_meta(str(abs_path))
            keyframe_id = Path(rel_path).stem
            doc_id = int.from_bytes(
                hashlib.md5(rel_path.encode()).digest()[:8], "big",
            )

            batch_points.append(PointStruct(
                id=doc_id,
                vector=embeddings[idx].tolist(),
                payload={
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "frame_idx": frame_idx,
                    "image_path": rel_path,
                },
            ))

            if len(batch_points) >= BATCH_SIZE:
                try:
                    client.upsert(collection_name=NEW_COLLECTION, points=batch_points, wait=True)
                except Exception as e:
                    log.error("  Upsert batch failed in %s: %s", folder.name, e)
                folder_upserted += len(batch_points)
                total_upserted += len(batch_points)
                batch_points = []

        # Flush remainder
        if batch_points:
            try:
                client.upsert(collection_name=NEW_COLLECTION, points=batch_points, wait=True)
                folder_upserted += len(batch_points)
                total_upserted += len(batch_points)
            except Exception as e:
                log.error("  Upsert final batch failed in %s: %s", folder.name, e)

        total_skipped += (N - folder_upserted)
        pbar.set_postfix_str(f"{folder.name} | up={total_upserted}/{total_vectors_expected}")

    pbar.close()

    # 3) Verify
    new_count = count_points(client, NEW_COLLECTION)
    log.info("=" * 70)
    log.info("[DONE] Collection '%s' has %d points (expected ~%d).", NEW_COLLECTION, new_count, total_vectors_expected)
    if old_count > 0:
        log.info("       vs OLD collection '%s': %d points -> delta=%d (re-indexed all? %s)",
                 OLD_COLLECTION, old_count, new_count - old_count,
                 "YES" if abs(new_count - old_count) < 10 else "NO (check missing!)")

    info = client.get_collection(NEW_COLLECTION)
    vec = info.config.params.vectors
    log.info("  Distance       : %s", vec.distance)
    log.info("  Dim            : %s", vec.size)
    try:
        log.info("  on_disk        : %s", vec.on_disk)
    except AttributeError:
        pass
    hnsw = info.config.hnsw_config
    log.info("  HNSW m/ef_cons : m=%s / ef_construct=%s", hnsw.m, hnsw.ef_construct)
    try:
        q = info.config.quantization_config
        log.info("  Quantization   : %s (always_ram=%s)",
                 getattr(q, "type", str(type(q))),
                 getattr(getattr(q, "scalar", None), "always_ram", None),
        )
    except Exception:
        pass
    log.info("  Segments count : %d", info.segments_count)
    log.info("=" * 70)
    log.info("[NEXT STEP] Now edit search/qdrant_search.py and set:")
    log.info("           COLLECTION_SIGLIP = '%s'", NEW_COLLECTION)
    log.info("           Or keep OLD, rename OLD to backup and NEW -> OLD (atom swap).")
    log.info("           Then restart backend (uvicorn main:app).")


if __name__ == "__main__":
    main()
