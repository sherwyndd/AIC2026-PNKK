#!/usr/bin/env python3
"""Stage 7: Build/Index keyframe SigLIP2 embeddings (.npy) into Qdrant Collection (FLOAT32 Full Recall)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.qdrant_builder import QdrantBuilder
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Stage 7: Qdrant Vector Indexing (Full Recall FLOAT32)")
    parser.add_argument("--qdrant_host", type=str, default="127.0.0.1", help="Qdrant server host")
    parser.add_argument("--qdrant_port", type=int, default=6333, help="Qdrant server port")
    parser.add_argument("--collection_name", type=str, default="image_siglip", help="Qdrant collection name")
    parser.add_argument("--vector_dim", type=int, default=1152, help="Vector dimension (1152 for SigLIP2 SO400M)")
    parser.add_argument("--embedding_dir", type=str, default=None, help="Directory containing .npy embedding files")
    parser.add_argument("--keyframe_dir", type=str, default=None, help="Directory containing keyframe image folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Pipeline output directory")
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--batch_size", type=int, default=500, help="Qdrant upsert batch size")
    parser.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection if exists")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    defaults = get_pipeline_defaults(config)
    mode = args.mode or defaults["mode"]

    output_dir = Path(args.output_dir or get_output_dir(mode, None, config))
    logger = setup_logger(log_file=str(output_dir / "pipeline.log"))

    logger.info(
        f"Starting Stage 7 Qdrant Vector Indexing "
        f"(Collection: '{args.collection_name}' @ {args.qdrant_host}:{args.qdrant_port}, dim={args.vector_dim}, Full Recall)..."
    )

    builder = QdrantBuilder(
        host=args.qdrant_host,
        port=args.qdrant_port,
        collection_name=args.collection_name,
        vector_dim=args.vector_dim,
    )

    if args.recreate:
        builder.create_collection(force_recreate=True)

    emb_dir = Path(args.embedding_dir or (output_dir / "embeddings"))
    kf_dir = Path(args.keyframe_dir or (output_dir / "keyframes"))

    if not emb_dir.exists():
        logger.error(f"Embedding directory not found: {emb_dir}")
        sys.exit(1)

    logger.info(f"Indexing .npy vectors from {emb_dir} into Qdrant collection '{args.collection_name}'...")
    total_indexed = builder.index_npy_files(
        embedding_dir=emb_dir,
        keyframe_root=kf_dir,
        batch_size=args.batch_size,
        logger=logger,
    )

    logger.info(
        f"Stage 7 Qdrant Vector Indexing completed. Successfully indexed {total_indexed} points into '{args.collection_name}'."
    )


if __name__ == "__main__":
    main()
