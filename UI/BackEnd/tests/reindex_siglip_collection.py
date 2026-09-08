"""Re-index Qdrant collection with SigLIP2-base embeddings (Multi-threaded NAS loading + Resume support).

⚠️  IMPORTANT: Uses encode_images_siglip() from models.siglip_encoder which guarantees
    the SAME feature extraction logic as encode_text_siglip(). This is critical for
    cross-modal alignment (text ↔ image must share the same embedding space).
"""

import os
import time
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from models.siglip_encoder import load_siglip, encode_images_siglip

# Config
KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
COLLECTION_NAME = "image_siglip"
BATCH_SIZE = 128
NUM_IO_WORKERS = 32
DEVICE = "cuda:0"
# ⚠️ Set RESUME=False if old collection was indexed with DIFFERENT extraction logic
#    (e.g. using pooler_output instead of image_embeds). Mixed vectors = bad search.
RESUME = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_image(path: str) -> Tuple[Image.Image, bool]:
    """Helper to read image in worker thread."""
    try:
        img = Image.open(path).convert("RGB")
        return img, True
    except Exception as err:
        logger.error("Error reading %s: %s", path, err)
        return None, False


def encode_images_batch(image_paths: List[str], io_pool: ThreadPoolExecutor) -> Tuple[np.ndarray, List[int]]:
    """Encode a batch of image paths with SigLIP2 using parallel disk reading.

    Uses encode_images_siglip() to guarantee identical feature extraction as text.
    """
    read_results = list(io_pool.map(_read_image, image_paths))

    images = []
    valid_indices = []
    for idx, (img, ok) in enumerate(read_results):
        if ok and img is not None:
            images.append(img)
            valid_indices.append(idx)

    if not images:
        return np.zeros((0, 768), dtype=np.float32), valid_indices

    # ✅ Use the unified encoder — same logic as encode_text_siglip()
    image_embs = encode_images_siglip(images)
    return image_embs, valid_indices


def get_all_keyframes() -> List[Path]:
    """Get all keyframe paths."""
    keyframes = []
    keyframe_dir = Path(KEYFRAME_ROOT)
    for video_dir in sorted(keyframe_dir.iterdir()):
        if video_dir.is_dir():
            keyframes.extend(sorted(video_dir.glob("*.jpg")))
    return keyframes


def main():
    logger.info("Loading SigLIP2 model on %s ...", DEVICE)
    load_siglip(device=DEVICE)

    client = QdrantClient(host="localhost", port=6333)

    existing_count = 0
    if RESUME:
        try:
            info = client.get_collection(COLLECTION_NAME)
            existing_count = info.points_count
            logger.info("Collection '%s' exists with %d points. Will resume from index %d.", COLLECTION_NAME, existing_count, existing_count)
            distance_used = info.config.params.vectors.distance
            if distance_used != Distance.COSINE:
                logger.warning("⚠️  Collection uses distance=%s but we recommend COSINE for normalized vectors.", distance_used)
        except Exception:
            logger.info("Collection '%s' does not exist. Creating new collection...", COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
    else:
        logger.info("Re-creating collection '%s' from scratch (deleting old one)...", COLLECTION_NAME)
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Old collection deleted successfully.")
        except Exception as e:
            logger.warning("Could not delete collection (may not exist): %s", e)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        logger.info("✅ New collection created with distance=COSINE for L2-normalized vectors.")

    logger.info("Scanning keyframes in %s ...", KEYFRAME_ROOT)
    keyframes = get_all_keyframes()
    total_keyframes = len(keyframes)
    logger.info("Found %d keyframes.", total_keyframes)

    start_idx = existing_count
    if start_idx >= total_keyframes:
        logger.info("Collection already has %d vectors (>= total keyframes %d). Complete!", existing_count, total_keyframes)
        return

    logger.info("Starting indexing from %d to %d with %d IO workers (Batch Size=%d)...", start_idx, total_keyframes, NUM_IO_WORKERS, BATCH_SIZE)

    point_id_counter = start_idx
    io_pool = ThreadPoolExecutor(max_workers=NUM_IO_WORKERS)

    try:
        for i in tqdm(range(start_idx, total_keyframes, BATCH_SIZE), desc="Encoding SigLIP2 Keyframes"):
            batch_paths = keyframes[i : i + BATCH_SIZE]
            batch_str_paths = [str(p) for p in batch_paths]

            embeddings, valid_indices = encode_images_batch(batch_str_paths, io_pool)

            points = []
            for emb_idx, orig_idx in enumerate(valid_indices):
                img_path = batch_paths[orig_idx]
                rel_path = img_path.relative_to(KEYFRAME_ROOT)
                video_id = rel_path.parts[0]
                try:
                    frame_idx = int(rel_path.stem.split("_kf_")[1])
                except Exception:
                    frame_idx = 0
                keyframe_id = rel_path.stem

                point = PointStruct(
                    id=point_id_counter,
                    vector=embeddings[emb_idx].tolist(),
                    payload={
                        "video_id": video_id,
                        "keyframe_id": keyframe_id,
                        "frame_idx": frame_idx,
                        "image_path": str(rel_path),
                    },
                )
                points.append(point)
                point_id_counter += 1

            if points:
                client.upsert(collection_name=COLLECTION_NAME, points=points)

    finally:
        io_pool.shutdown(wait=False)

    logger.info("Re-indexing completed! Total indexed points: %d", point_id_counter)


if __name__ == "__main__":
    main()
