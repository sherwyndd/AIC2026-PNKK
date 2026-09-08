#!/usr/bin/env python3
"""Stage 5: Keyframe Image Feature Vector Extraction using SigLIP2."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.siglip2_embedder import SigLIP2Embedder
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Stage 5: SigLIP2 Keyframe Image Vector Extraction")
    parser.add_argument("--keyframe_dir", type=str, default=None, help="Directory containing extracted keyframes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for .npy embeddings")
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for SigLIP2 inference")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    defaults = get_pipeline_defaults(config)
    mode = args.mode or defaults["mode"]

    output_dir = Path(args.output_dir or get_output_dir(mode, None, config))
    emb_out_dir = output_dir / "embeddings"
    emb_out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=str(output_dir / "pipeline.log"))
    logger.info("Starting Stage 5 SigLIP2 Keyframe Image Feature Extraction...")

    kf_root = Path(args.keyframe_dir or (output_dir / "keyframes"))
    if not kf_root.exists():
        logger.error(f"Keyframe root directory not found: {kf_root}")
        sys.exit(1)

    video_dirs = sorted([d for d in kf_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(video_dirs)} video keyframe folder(s) in {kf_root}")

    embedder = SigLIP2Embedder()

    processed = 0
    for vdir in video_dirs:
        npy_path = emb_out_dir / f"{vdir.name}.npy"
        if npy_path.exists() and npy_path.stat().st_size > 0:
            logger.info(f"Skipping embedding for {vdir.name}: already exists at {npy_path}")
            continue

        logger.info(f"Generating SigLIP2 embeddings for video: {vdir.name}")
        matrix = embedder.embed_keyframe_dir(
            keyframe_dir=vdir,
            output_npy=npy_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
        )
        if matrix is not None:
            processed += 1

    logger.info(f"Stage 5 Embedding completed. Generated {processed} .npy file(s) in {emb_out_dir}")


if __name__ == "__main__":
    main()
