#!/usr/bin/env python3
"""Master pipeline runner: Orchestrates the 7-stage preprocessing & indexing workflow.

Stages:
  1. shot_cut           : TransNetV2 shot boundary detection
  2. keyframe_ext       : Candidate sampling & CLIP relative diff filtering
  3. ocr                : PaddleOCR text extraction from keyframes
  4. asr_whisper        : Faster-Whisper ASR speech transcription & 6s window mapping
  5. embedding          : SigLIP2 image vector feature extraction (.npy)
  6. build_elasticsearch: Text metadata indexing into Elasticsearch
  7. build_qdrant       : Scalar Quantization (SQ8) vector indexing into Qdrant
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_layout, get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.keyframe_ext import run_keyframe_ext
from src.pipeline.shot_cut import run_shot_cut
from src.pipeline.ocr_engine import OCREngine
from src.pipeline.siglip2_embedder import SigLIP2Embedder
from src.pipeline.elasticsearch_builder import ElasticsearchBuilder
from src.pipeline.qdrant_builder import QdrantBuilder
from src.utils.gpu_utils import free_vram
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="7-Stage Preprocessing & Indexing Pipeline")
    parser.add_argument("--input_dir", type=str, default=str(defaults["input_dir"]))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--format", type=str, choices=["csv", "json"], default=defaults["format"])
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run (e.g., '1,2,3,4,5,6,7', 'all', or 'shot_cut,keyframe_ext,ocr')",
    )
    parser.add_argument("--sample_step", type=int, default=defaults["sample_step"])
    parser.add_argument("--rel_diff_threshold", type=float, default=defaults["rel_diff_threshold"])
    parser.add_argument("--clip_batch_size", type=int, default=defaults["clip_batch_size"])
    parser.add_argument("--ocr_lang", type=str, default="vi", help="Language for OCR stage")
    parser.add_argument("--asr_language", type=str, default="vi", help="Language for ASR stage")
    parser.add_argument("--asr_beam_size", type=int, default=5)
    parser.add_argument("--asr_window", type=float, default=3.0)
    parser.add_argument("--asr_model_size", type=str, default="medium")
    parser.add_argument("--es_url", type=str, default="http://127.0.0.1:9200")
    parser.add_argument("--es_index", type=str, default="aic2026_elastics_text")
    parser.add_argument("--qdrant_host", type=str, default="127.0.0.1")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    parser.add_argument("--qdrant_collection", type=str, default="image_siglip")
    parser.add_argument("--force", action="store_true", help="Rerun stages even if log indicates completion")
    parser.add_argument("--video_list", type=str, default=None, help="Path to text file listing video filenames")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/runtime.yaml")
    return parser


def parse_selected_stages(stages_arg: str) -> set:
    if stages_arg.strip().lower() == "all":
        return {1, 2, 3, 4, 5, 6, 7}

    stage_map = {
        "shot_cut": 1,
        "keyframe_ext": 2,
        "ocr": 3,
        "asr_whisper": 4,
        "embedding": 5,
        "build_elasticsearch": 6,
        "build_qdrant": 7,
    }

    selected = set()
    tokens = [t.strip() for t in stages_arg.split(",") if t.strip()]
    for token in tokens:
        if token.isdigit():
            selected.add(int(token))
        elif token.lower() in stage_map:
            selected.add(stage_map[token.lower()])
    return selected


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    defaults = get_pipeline_defaults(config)
    mode = args.mode or defaults["mode"]
    layout = get_layout(mode)

    output_dir = str(get_output_dir(mode, args.output_dir, config))
    input_dir = args.input_dir or str(defaults["input_dir"])
    out_format = args.format or defaults["format"]
    log_file = Path(output_dir) / "pipeline.log"
    logger = setup_logger(log_file=str(log_file))

    selected_stages = parse_selected_stages(args.stages)

    logger.info("==================================================")
    logger.info("   7-Stage Preprocessing & Indexing Pipeline")
    logger.info("==================================================")
    logger.info(f"Input dir  : {input_dir}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Mode       : {mode}")
    logger.info(f"Selected   : Stages {sorted(list(selected_stages))}")
    logger.info("==================================================")

    # Stage 1: Shot cut
    if 1 in selected_stages:
        logger.info("\n--- Stage 1: Shot boundary detection (TransNetV2) ---")
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

    # Stage 2: Keyframe extraction
    if 2 in selected_stages:
        logger.info("\n--- Stage 2: Keyframe extraction & CLIP filtering ---")
        run_keyframe_ext(
            input_dir=input_dir,
            output_dir=output_dir,
            mode=mode,
            out_format=out_format,
            sample_step=args.sample_step,
            rel_diff_threshold=args.rel_diff_threshold,
            clip_batch_size=args.clip_batch_size,
            clip_model_id=defaults["clip_model_id"],
            video_list_path=args.video_list,
            layout=layout,
            logger=logger,
            force=args.force,
            log_file=str(log_file),
        )

    # Stage 3: OCR
    if 3 in selected_stages:
        logger.info("\n--- Stage 3: OCR text extraction (PaddleOCR) ---")
        try:
            kf_root = Path(output_dir) / "keyframes"
            ocr_out_dir = Path(output_dir) / "ocr"
            ocr_out_dir.mkdir(parents=True, exist_ok=True)
            if kf_root.exists():
                engine = OCREngine(lang=args.ocr_lang)
                vdirs = sorted([d for d in kf_root.iterdir() if d.is_dir()])
                for vdir in vdirs:
                    out_f = ocr_out_dir / f"{vdir.name}.json"
                    if not out_f.exists() or args.force:
                        engine.process_keyframe_dir(vdir, out_f, logger=logger)
        except Exception as exc:
            logger.error(f"Stage 3 OCR failed: {exc}")

    # Stage 4: ASR Whisper
    if 4 in selected_stages:
        logger.info("\n--- Stage 4: ASR speech transcription (Whisper) ---")
        try:
            from src.pipeline.asr import run_asr_whisper
            run_asr_whisper(
                input_dir=input_dir,
                output_dir=output_dir,
                mode=mode,
                asr_format="json",
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
            logger.error(f"Stage 4 ASR failed: {exc}")

    # Stage 5: Embedding SigLIP2
    if 5 in selected_stages:
        logger.info("\n--- Stage 5: SigLIP2 keyframe vector embedding ---")
        try:
            kf_root = Path(output_dir) / "keyframes"
            emb_dir = Path(output_dir) / "embeddings"
            emb_dir.mkdir(parents=True, exist_ok=True)
            if kf_root.exists():
                embedder = SigLIP2Embedder()
                vdirs = sorted([d for d in kf_root.iterdir() if d.is_dir()])
                for vdir in vdirs:
                    npy_f = emb_dir / f"{vdir.name}.npy"
                    if not npy_f.exists() or args.force:
                        embedder.embed_keyframe_dir(vdir, npy_f, logger=logger)
        except Exception as exc:
            logger.error(f"Stage 5 Embedding failed: {exc}")

    # Stage 6: Build Elasticsearch
    if 6 in selected_stages:
        logger.info("\n--- Stage 6: Elasticsearch text indexing ---")
        try:
            es_builder = ElasticsearchBuilder(es_url=args.es_url, index_name=args.es_index)
            ocr_dir = Path(output_dir) / "ocr"
            asr_dir = Path(output_dir) / "metadata" / "asr"
            if not asr_dir.exists():
                asr_dir = Path(output_dir) / "asr"

            docs = {}
            for of in (list(ocr_dir.glob("*.json")) if ocr_dir.exists() else []):
                for r in es_builder.load_json_records(of):
                    img = str(r.get("image", ""))
                    kf_id = Path(img).stem
                    if kf_id:
                        docs[kf_id] = {
                            "keyframe_id": kf_id,
                            "video_id": of.stem,
                            "image_path": img,
                            "ocr_text": str(r.get("text", "")),
                            "asr_text": "",
                            "caption_text": "",
                            "object_text": "",
                            "dense_ocr_text": "",
                            "merged_text": str(r.get("text", "")),
                        }

            for af in (list(asr_dir.glob("*.json*")) if asr_dir.exists() else []):
                for r in es_builder.load_json_records(af):
                    kf_id = str(r.get("keyframe_id", ""))
                    if not kf_id:
                        continue
                    asr_txt = ""
                    meta = r.get("asr_metadata", {})
                    if isinstance(meta, dict):
                        asr_txt = str(meta.get("asr_text_6s", ""))
                    if kf_id in docs:
                        docs[kf_id]["asr_text"] = asr_txt
                        docs[kf_id]["merged_text"] += f" {asr_txt}"
                    else:
                        docs[kf_id] = {
                            "keyframe_id": kf_id,
                            "video_id": str(r.get("video_id", af.stem.replace("_asr", ""))),
                            "image_path": str(r.get("image_path", "")),
                            "ocr_text": "",
                            "asr_text": asr_txt,
                            "caption_text": "",
                            "object_text": "",
                            "dense_ocr_text": "",
                            "merged_text": asr_txt,
                        }

            if docs:
                count = es_builder.bulk_import_documents(docs.values())
                logger.info(f"Indexed {count} text documents to Elasticsearch index '{args.es_index}'")
        except Exception as exc:
            logger.error(f"Stage 6 Elasticsearch indexing failed: {exc}")

    # Stage 7: Build Qdrant
    if 7 in selected_stages:
        logger.info("\n--- Stage 7: Qdrant SQ8 vector indexing ---")
        try:
            qd_builder = QdrantBuilder(
                host=args.qdrant_host,
                port=args.qdrant_port,
                collection_name=args.qdrant_collection,
            )
            emb_dir = Path(output_dir) / "embeddings"
            kf_dir = Path(output_dir) / "keyframes"
            if emb_dir.exists():
                count = qd_builder.index_npy_files(emb_dir, kf_dir, logger=logger)
                logger.info(f"Indexed {count} vector points to Qdrant collection '{args.qdrant_collection}'")
        except Exception as exc:
            logger.error(f"Stage 7 Qdrant indexing failed: {exc}")

    free_vram()
    logger.info("\n==================================================")
    logger.info("   Pipeline Execution Finished Successfully!")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
