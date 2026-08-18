#!/usr/bin/env python3
"""Master script: run shot cut then keyframe extraction."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.keyframe_ext import run_keyframe_ext
from src.pipeline.shot_cut import run_shot_cut
from src.utils.gpu_utils import free_vram
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Two-stage keyframe preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, default=str(defaults["input_dir"]))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--format", type=str, choices=["csv", "json"], default=defaults["format"])
    parser.add_argument("--sample_step", type=int, default=defaults["sample_step"])
    parser.add_argument("--rel_diff_threshold", type=float, default=defaults["rel_diff_threshold"])
    parser.add_argument("--clip_batch_size", type=int, default=defaults["clip_batch_size"])
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "shot_cut", "keyframe_ext", "asr_whisper", "object_detection", "captioning"],
        default="all",
        help="Run all stages or a single stage",
    )
    parser.add_argument("--shots_dir", type=str, default=None, help="Override shots metadata directory")
    parser.add_argument("--yolov8_model_id", type=str, default=defaults["yolov8_model_id"])
    parser.add_argument("--det_conf_threshold", type=float, default=defaults["object_detection_confidence"])
    parser.add_argument("--det_iou_threshold", type=float, default=defaults["object_detection_iou"])
    parser.add_argument("--caption_model_id", type=str, default=defaults["caption_model_id"])
    parser.add_argument("--caption_prompt", type=str, default=defaults["caption_prompt"])
    parser.add_argument("--force", action="store_true", help="Rerun stages even if pipeline.log indicates completion")
    parser.add_argument("--asr_format", type=str, choices=["json", "jsonl"], default="json")
    parser.add_argument("--asr_language", type=str, default="vi")
    parser.add_argument("--asr_beam_size", type=int, default=5)
    parser.add_argument("--asr_window", type=float, default=3.0)
    parser.add_argument("--asr_model_size", type=str, default="medium")
    parser.add_argument("--video_list", type=str, default=None, help="Path to text file listing video filenames to selectively process for stages such as shot cut, keyframe extraction, and object detection")
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

    input_dir = args.input_dir or str(defaults["input_dir"])
    out_format = args.format or defaults["format"]
    log_file = Path(output_dir) / "pipeline.log"
    logger = setup_logger(log_file=str(log_file))

    logger.info(f"input_dir:  {input_dir}")
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"mode:       {mode}")
    logger.info(f"stage:      {args.stage}")

    if args.stage in ("all", "shot_cut"):
        logger.info("=== Stage 1: Shot cut ===")
        run_shot_cut(
            input_dir=input_dir,
            output_dir=output_dir,
            mode=mode,
            out_format=out_format,
            video_list_path=args.video_list,
            layout=layout,
            logger=logger,
            force=args.force,
            log_file=str(log_file),
        )

    if args.stage in ("all", "keyframe_ext"):
        logger.info("=== Stage 2: Keyframe extraction ===")
        run_keyframe_ext(
            input_dir=input_dir,
            output_dir=output_dir,
            mode=mode,
            out_format=out_format,
            sample_step=args.sample_step,
            rel_diff_threshold=args.rel_diff_threshold,
            clip_batch_size=args.clip_batch_size,
            clip_model_id=defaults["clip_model_id"],
            shots_dir=args.shots_dir if hasattr(args, 'shots_dir') else None,
            video_list_path=args.video_list,
            layout=layout,
            logger=logger,
            force=args.force,
            log_file=str(log_file),
        )

    if args.stage in ("all", "asr_whisper"):
        from src.pipeline.asr import run_asr_whisper

        logger.info("=== Stage 3: ASR whisper ===")
        try:
            run_asr_whisper(
                input_dir=input_dir,
                output_dir=output_dir,
                mode=mode,
                asr_format=args.asr_format,
                data_format=out_format,
                language=args.asr_language,
                beam_size=args.asr_beam_size,
                window_sec=args.asr_window,
                model_size=args.asr_model_size,
                layout=layout,
                logger=logger,
                force=args.force,
                log_file=str(log_file),
                video_list_path=args.video_list,
            )
        except Exception as exc:
            logger.error(f"ASR whisper stage failed: {exc}")
            logger.error(f"[PIPELINE] stage=asr_whisper status=stage_failed")

    if args.stage in ("all", "object_detection"):
        from src.pipeline.object_detection import run_object_detection

        logger.info("=== Stage 4: Object detection ===")
        try:
            run_object_detection(
                input_dir=input_dir,
                output_dir=output_dir,
                mode=mode,
                out_format=out_format,
                yolov8_model_id=args.yolov8_model_id,
                conf_threshold=args.det_conf_threshold,
                iou_threshold=args.det_iou_threshold,
                layout=layout,
                logger=logger,
                force=args.force,
                log_file=str(log_file),
                video_list_path=args.video_list,
            )
        except Exception as exc:
            logger.error(f"Object detection stage failed: {exc}")
            logger.error(f"[PIPELINE] stage=object_detection status=stage_failed")

    if args.stage in ("all", "captioning"):
        from src.pipeline.captioning import run_captioning

        logger.info("=== Stage 5: Captioning ===")
        try:
            run_captioning(
                input_dir=input_dir,
                output_dir=output_dir,
                mode=mode,
                out_format=out_format,
                caption_model_id=args.caption_model_id,
                caption_prompt=args.caption_prompt,
                layout=layout,
                logger=logger,
                force=args.force,
                log_file=str(log_file),
                video_list_path=args.video_list,
            )
        except Exception as exc:
            logger.error(f"Captioning stage failed: {exc}")
            logger.error(f"[PIPELINE] stage=captioning status=stage_failed")

    free_vram()
    logger.info(f"Pipeline complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
