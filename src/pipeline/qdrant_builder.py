import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def parse_image_meta(image_name: str) -> Tuple[str, int]:
    """Parse video_id and frame_idx from image filename like L01_V001_kf_0012."""
    stem = Path(image_name).stem
    m = re.match(r"(L\d+_V\d+)_kf_(\d+)", stem)
    if m is None:
        return stem, 0
    return m.group(1), int(m.group(2))


def generate_point_id(keyframe_id: str) -> str:
    """Generate deterministic UUID-like string from keyframe_id string."""
    return hashlib.md5(keyframe_id.encode("utf-8")).hexdigest()


class QdrantBuilder:
    """Core Qdrant Collection & Vector Index Builder matching full-recall FLOAT32 configuration.
    
    Collection settings match reindex_siglip_full_recall.py:
      - dim: 1152 (google/siglip2-so400m-patch14-384)
      - distance: COSINE
      - quantization: None (FLOAT32 full precision for max recall)
      - HNSW: m=32, ef_construct=512, full_scan_threshold=20000
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        collection_name: str = "image_siglip",
        vector_dim: int = 1152,
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(host=self.host, port=self.port, timeout=120.0)

    def create_collection(self, force_recreate: bool = False):
        collections = [c.name for c in self.client.get_collections().collections]
        if force_recreate and self.collection_name in collections:
            self.client.delete_collection(self.collection_name)
            _logger.info(f"Deleted existing Qdrant collection: {self.collection_name}")
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                    on_disk=False,  # RAM-only for high-speed retrieval
                ),
                quantization_config=None,  # No quantization for maximum recall
                hnsw_config=HnswConfigDiff(
                    m=32,
                    ef_construct=512,
                    full_scan_threshold=20000,
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                ),
            )
            _logger.info(
                f"Created Qdrant collection '{self.collection_name}' (dim={self.vector_dim}, FLOAT32, HNSW m=32 ef_construct=512)"
            )

    def index_npy_files(
        self,
        embedding_dir: Path,
        keyframe_root: Path,
        batch_size: int = 500,
        logger: Optional[Any] = None,
    ) -> int:
        """Scan .npy embedding files and upload vector points with payload into Qdrant."""
        self.create_collection(force_recreate=False)
        npy_files = sorted(Path(embedding_dir).glob("*.npy"))
        if not npy_files:
            if logger:
                logger.warning(f"No .npy embedding files found in {embedding_dir}")
            return 0

        total_points = 0
        points_batch: List[PointStruct] = []

        for npy_path in tqdm(npy_files, desc="Indexing Qdrant (Full Recall)"):
            try:
                matrix = np.load(npy_path)
            except Exception as exc:
                if logger:
                    logger.error(f"Error loading {npy_path}: {exc}")
                continue

            # Auto-detect vector dimension if matrix has data
            if matrix.ndim == 2 and matrix.shape[1] != self.vector_dim:
                self.vector_dim = matrix.shape[1]

            video_id = npy_path.stem
            kf_dir = keyframe_root / video_id
            images = sorted(kf_dir.glob("*.jpg")) if kf_dir.exists() else []

            for idx, vec in enumerate(matrix):
                if idx < len(images):
                    img_path = str(images[idx])
                    kf_id = images[idx].stem
                else:
                    kf_id = f"{video_id}_kf_{idx+1:04d}"
                    img_path = f"keyframes/{video_id}/{kf_id}.jpg"

                vid, frame_idx = parse_image_meta(kf_id)
                point_id = generate_point_id(kf_id)

                payload = {
                    "keyframe_id": kf_id,
                    "video_id": vid,
                    "frame_idx": frame_idx,
                    "image_path": img_path,
                }

                points_batch.append(
                    PointStruct(
                        id=point_id,
                        vector=vec.tolist(),
                        payload=payload,
                    )
                )

                if len(points_batch) >= batch_size:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_batch,
                    )
                    total_points += len(points_batch)
                    points_batch = []

        if points_batch:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points_batch,
            )
            total_points += len(points_batch)

        if logger:
            logger.info(
                f"Successfully indexed {total_points} vector points into Qdrant collection '{self.collection_name}'"
            )

        return total_points
