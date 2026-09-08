"""Re-create SigLIP collection toi uu RECALL (KHONG quantization).

REUSES pre-computed .npy embeddings from nhan's pipeline (no re-encoding! fast).
Creates a NEW collection 'image_siglip' — full FLOAT32, HNSW chat luong cao.

[UPGRADE 31/08] Doi sang siglip2_final embeddings (so400m-patch14-384, dim=1152):
  - EMBEDDING_DIR: outputs/siglip2_final/ (thay vi outputs/embeddings/)
  - dim: 768 -> 1152 (model lon hon, recall tot hon)
  - Image size: 224 -> 384 (patch14 thay vi patch16)

Vi sao bo quantization:
  - RAM du da (488GB free), toc do hien tai da du nhanh (nho storage o /dev/shm = RAM)
  - Bo INT8 quantization loai hoan toan sai so nen -> recall toi da tuyet doi
  - Don gian hoa pipeline search (khong can lo rescore/oversampling)

Config applied:
  - Vectors:      dim=1152, distance=COSINE, on_disk=false (RAM only)
  - Model source: google/siglip2-so400m-patch14-384
  - Quantization: KHONG DUNG (bo han) -> recall toi da, khong mat thong tin do nen
  - HNSW:         m=32, ef_construct=512
                  -> đồ thị dày hơn, nhiều "hàng xóm" hơn mỗi node -> recall cao hơn
  - full_scan_threshold=20000 (tăng từ 10000)
                  -> khi filter theo video_list thu hẹp candidate xuống dưới 20k,
                     Qdrant full-scan chính xác 100% thay vì dùng HNSW xấp xỉ
  - Optimizer:    indexing_threshold=20000 (giữ nguyên, build index sớm)

Đánh đổi:
  - RAM dung gap ~4x so voi ban SQ8 (do khong nen INT8) -> van rat nho so voi 488GB
    Vi du: 1,000,000 vectors x 1152 dim x 4 bytes = ~4.6GB (khong dang ke vs 488GB)
  - Build lau hon 1 chut do ef_construct/m cao hon (chi anh huong luc build, 1 lan)
  - Search có thể chậm hơn SQ8 một chút (do không có bước lọc INT8 nhanh), nhưng
    bù lại recall tối đa và code search đơn giản hơn hẳn

Usage:
    cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd
    python3 tests/reindex_siglip_full_recall.py
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
    HnswConfigDiff,
    OptimizersConfigDiff,
)

# —— Paths (reuse existing .npy) ————————————————————————————————————————————
EMBEDDING_DIR = Path("/AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/siglip2_final")
KEYFRAME_ROOT = Path("/AIClub_NAS/core_baotg/nhan/dataset/keyframes")
NEW_COLLECTION = "image_siglip"
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fp32_hq_import")


def parse_image_meta(image_path: str):
    stem = Path(image_path).stem
    m = re.match(r"(L\d+_V\d+)_kf_(\d+)", stem)
    if m is None:
        log.warning("Cannot parse filename (fallback): %s", stem)
        return stem, 0
    return m.group(1), int(m.group(2))


def create_fp32_hq_collection(client: QdrantClient):
    """Create image_siglip: FLOAT32, no quantization, high-quality HNSW."""
    cols = [c.name for c in client.get_collections().collections]
    if NEW_COLLECTION in cols:
        log.warning("Collection %s already exists -> deleting first to start clean.", NEW_COLLECTION)
        client.delete_collection(NEW_COLLECTION)

    log.info(
        "Creating collection '%s' | dim=1152 COSINE | NO quantization (FLOAT32 full) | "
        "HNSW m=32 ef_construct=512 | full_scan_threshold=20000 | on_disk=false",
        NEW_COLLECTION,
    )

    # Configs
    vectors_cfg = VectorParams(
        size=1152,
        distance=Distance.COSINE,
        on_disk=False,               # RAM only for speed (488GB free, plenty of headroom)
    )
    hnsw_cfg = HnswConfigDiff(
        m=32,                        # [UPGRADE] tăng từ 16 -> 32: mỗi node nhiều hàng xóm hơn
        ef_construct=512,            # [UPGRADE] tăng từ 256 -> 512: đồ thị build kỹ hơn
        full_scan_threshold=20000,   # [UPGRADE] tăng từ 10000: filter hẹp vẫn chính xác 100%
        on_disk=False,               # đảm bảo index HNSW cũng ở RAM, không chỉ vector
    )
    opt_cfg = OptimizersConfigDiff(
        indexing_threshold=20000,    # build index sớm trong lúc import
        memmap_threshold=1000000,    # segment > 1M điểm mới cân nhắc mmap (an toàn, dataset chưa tới)
    )

    # KHÔNG truyền quantization_config -> Qdrant giữ FLOAT32 thuần, không nén
    try:
        client.create_collection(
            collection_name=NEW_COLLECTION,
            vectors_config=vectors_cfg,
            hnsw_config=hnsw_cfg,
            optimizers_config=opt_cfg,
        )
        log.info("[OK] Collection '%s' created (FLOAT32, no quantization).", NEW_COLLECTION)
    except Exception as e:
        log.exception("[FATAL] Cannot create collection: %s", e)
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

    # 1) Create NEW FP32 high-quality collection
    create_fp32_hq_collection(client)

    # 2) Import vectors FROM .npy files (fast, no GPU encoding)
    dirs = list_video_dirs()

    total_vectors_expected = 0
    for d in dirs:
        with open(d / "image_paths.txt") as f:
            total_vectors_expected += sum(1 for line in f if line.strip())
    log.info("Total vectors to import: %d", total_vectors_expected)

    pbar = tqdm(dirs, desc="Import .npy -> image_siglip", unit="folder")
    total_upserted = 0
    total_skipped = 0

    for folder in pbar:
        pbar.set_postfix_str(folder.name)
        npy_file = folder / "image_embeddings.npy"
        txt_file = folder / "image_paths.txt"

        try:
            embeddings = np.load(npy_file).astype(np.float32)  # (N, 1152) — siglip2_final
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

    info = client.get_collection(NEW_COLLECTION)
    vec = info.config.params.vectors
    log.info("  Distance       : %s", vec.distance)
    log.info("  Dim            : %s", vec.size)
    try:
        log.info("  on_disk        : %s", vec.on_disk)
    except AttributeError:
        pass
    hnsw = info.config.hnsw_config
    log.info("  HNSW m/ef_cons : m=%s / ef_construct=%s / full_scan_threshold=%s",
              hnsw.m, hnsw.ef_construct, hnsw.full_scan_threshold)
    log.info("  Quantization   : NONE (full FLOAT32, recall toi da)")
    log.info("  Model source   : google/siglip2-so400m-patch14-384 (dim=1152)")
    log.info("  Segments count : %d", info.segments_count)
    log.info("=" * 70)
    log.info("[NEXT STEP] Now edit search/qdrant_search.py and set:")
    log.info("           COLLECTION_SIGLIP = '%s'", NEW_COLLECTION)
    log.info("           Đồng thời bỏ QuantizationSearchParams trong _build_search_params()")
    log.info("           (không còn quantization để rescore nữa).")
    log.info("           Then restart backend (uvicorn main:app).")


if __name__ == "__main__":
    main()