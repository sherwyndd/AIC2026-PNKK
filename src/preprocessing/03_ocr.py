#!/usr/bin/env python3
"""Stage 3: OCR text extraction from keyframes using PaddleOCR."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.ocr_engine import OCREngine
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Stage 3: Keyframe OCR Text Extraction")
    parser.add_argument("--keyframe_dir", type=str, default=None, help="Directory containing extracted keyframes")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for OCR JSON files")
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--lang", type=str, default="vi", help="OCR language")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for text lines")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for OCR inference")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    defaults = get_pipeline_defaults(config)
    mode = args.mode or defaults["mode"]
    layout = get_layout(mode)

    output_dir = Path(args.output_dir or get_output_dir(mode, None, config))
    ocr_out_dir = output_dir / "ocr"
    ocr_out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=str(output_dir / "pipeline.log"))
    logger.info(f"Starting Stage 3 OCR (language: {args.lang})...")

    kf_root = Path(args.keyframe_dir or (output_dir / "keyframes"))
    if not kf_root.exists():
        logger.error(f"Keyframe root directory not found: {kf_root}")
        sys.exit(1)

    video_dirs = sorted([d for d in kf_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(video_dirs)} video keyframe folder(s) in {kf_root}")

    try:
        engine = OCREngine(lang=args.lang)
    except Exception as exc:
        logger.error(f"Failed to initialize OCR engine: {exc}")
        sys.exit(1)

    processed = 0
    for vdir in video_dirs:
        out_file = ocr_out_dir / f"{vdir.name}.json"
        if out_file.exists() and out_file.stat().st_size > 0:
            logger.info(f"Skipping OCR for {vdir.name}: already exists at {out_file}")
            continue

        logger.info(f"Processing OCR for video: {vdir.name}")
        res = engine.process_keyframe_dir(
            keyframe_dir=vdir,
            output_file=out_file,
            conf_threshold=args.conf_threshold,
            batch_size=args.batch_size,
            logger=logger,
        )
        processed += 1

    logger.info(f"Stage 3 OCR completed. Processed {processed} video(s). Output in {ocr_out_dir}")


if __name__ == "__main__":
    main()
