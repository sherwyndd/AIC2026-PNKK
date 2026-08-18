#!/usr/bin/env python3
"""Stage 2: Candidate sampling + CLIP semantic keyframe filtering."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.keyframe_ext import run_keyframe_ext
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Keyframe extraction with CLIP L2 filtering")
    parser.add_argument("--input_dir", type=str, default=str(defaults["input_dir"]))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--shots_dir", type=str, default=None, help="Override shots metadata directory")
    parser.add_argument("--video_list", type=str, default=None, help="Path to text file listing video filenames to process")
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--format", type=str, choices=["csv", "json"], default=defaults["format"])
    parser.add_argument("--sample_step", type=int, default=defaults["sample_step"])
    parser.add_argument("--rel_diff_threshold", type=float, default=defaults["rel_diff_threshold"])
    parser.add_argument("--clip_batch_size", type=int, default=defaults["clip_batch_size"])
    parser.add_argument("--force", action="store_true", help="Rerun keyframe_ext even if pipeline.log indicates completion")
    parser.add_argument("--wait", action="store_true", default=True, help="Wait for Stage 1 shot_cut metadata if not ready for a video")
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Do not wait for Stage 1 shot_cut metadata, skip if missing")
    parser.add_argument("--config", type=str, default=None)
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

    result = run_keyframe_ext(
        input_dir=args.input_dir or str(defaults["input_dir"]),
        output_dir=output_dir,
        mode=mode,
        out_format=args.format or defaults["format"],
        sample_step=args.sample_step,
        rel_diff_threshold=args.rel_diff_threshold,
        clip_batch_size=args.clip_batch_size,
        clip_model_id=defaults["clip_model_id"],
        shots_dir=args.shots_dir,
        video_list_path=args.video_list,
        layout=layout,
        logger=logger,
        force=args.force,
        log_file=str(Path(output_dir) / "pipeline.log"),
        wait=args.wait,
    )
    logger.info(f"Keyframe extraction finished. Processed {result['processed_videos']} video(s).")
    logger.info(f"Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
