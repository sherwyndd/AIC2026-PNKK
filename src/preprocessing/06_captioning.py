#!/usr/bin/env python3
"""Stage 5: Image captioning on extracted keyframes with InternVL 2.5."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.captioning import run_captioning
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Caption keyframes using InternVL 2.5")
    parser.add_argument("--input_dir", type=str, default=str(defaults["input_dir"]))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--format", type=str, choices=["csv", "json", "jsonl"], default=defaults["format"])
    parser.add_argument("--caption_model_id", type=str, default=defaults["caption_model_id"])
    parser.add_argument(
        "--caption_prompt",
        type=str,
        default=defaults["caption_prompt"],
        help="Prompt used for the image captioning model",
    )
    parser.add_argument("--force", action="store_true", help="Rerun captioning even if pipeline.log indicates completion")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    defaults = get_pipeline_defaults(config)
    mode = args.mode or defaults["mode"]
    layout = get_layout(mode)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(get_output_dir(mode, None, config))
    logger = setup_logger(log_file=str(Path(output_dir) / "pipeline.log"))

    result = run_captioning(
        input_dir=args.input_dir or str(defaults["input_dir"]),
        output_dir=output_dir,
        mode=mode,
        out_format=args.format,
        caption_model_id=args.caption_model_id,
        caption_prompt=args.caption_prompt,
        layout=layout,
        logger=logger,
        force=args.force,
        log_file=str(Path(output_dir) / "pipeline.log"),
    )

    logger.info(f"Captioning finished. Processed {result['processed_videos']} video(s).")
    logger.info(f"Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
