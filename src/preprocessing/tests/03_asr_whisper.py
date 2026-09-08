#!/usr/bin/env python3
"""Legacy Stage 3: ASR extraction with faster-whisper (Archived in tests/ directory)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.asr import run_asr_whisper
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="ASR extraction for keyframe-based pipeline (Legacy Stage 3)")
    parser.add_argument("--input_dir", type=str, default=str(defaults["input_dir"]))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--asr_format", type=str, choices=["json", "jsonl"], default="json")
    parser.add_argument("--data_format", type=str, choices=["csv", "json", "jsonl"], default=defaults["format"])
    parser.add_argument("--asr_language", type=str, default="vi")
    parser.add_argument("--asr_beam_size", type=int, default=5)
    parser.add_argument("--asr_window", type=float, default=3.0)
    parser.add_argument("--asr_model_size", type=str, default="medium")
    parser.add_argument("--force", action="store_true", help="Rerun ASR even if pipeline.log indicates completion")
    parser.add_argument("--video_list", type=str, default=None, help="Path to text file listing video filenames to process")
    parser.add_argument("--wait", action="store_true", default=True, help="Wait for Stage 2 keyframes metadata if not ready for a video")
    parser.add_argument("--no-wait", action="store_false", dest="wait", help="Do not wait for Stage 2 keyframes metadata, skip if missing")
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

    result = run_asr_whisper(
        input_dir=args.input_dir or str(defaults["input_dir"]),
        output_dir=output_dir,
        mode=mode,
        asr_format=args.asr_format,
        data_format=args.data_format,
        language=args.asr_language,
        beam_size=args.asr_beam_size,
        window_sec=args.asr_window,
        model_size=args.asr_model_size,
        layout=layout,
        logger=logger,
        force=args.force,
        log_file=str(Path(output_dir) / "pipeline.log"),
        video_list_path=args.video_list,
        wait=args.wait,
    )

    logger.info(f"ASR whisper finished. Processed {result['processed_videos']} video(s).")
    logger.info(f"Output: {result['output_dir']}")


if __name__ == "__main__":
    main()
