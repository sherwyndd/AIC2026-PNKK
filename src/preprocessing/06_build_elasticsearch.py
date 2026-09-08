#!/usr/bin/env python3
"""Stage 6: Build/Index keyframe text metadata (OCR, ASR, Captions, Objects) into Elasticsearch."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_output_dir, get_pipeline_defaults, load_runtime_config
from src.pipeline.elasticsearch_builder import ElasticsearchBuilder
from src.utils.logger import setup_logger


def build_parser():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Stage 6: Elasticsearch Text Metadata Indexing")
    parser.add_argument("--es_url", type=str, default="http://127.0.0.1:9200", help="Elasticsearch server URL")
    parser.add_argument("--index_name", type=str, default="aic2026_elastics_text", help="Elasticsearch index name")
    parser.add_argument("--output_dir", type=str, default=None, help="Pipeline output directory containing metadata")
    parser.add_argument("--mode", type=str, choices=["full", "demo"], default=defaults["mode"])
    parser.add_argument("--batch_size", type=int, default=1000, help="Bulk index chunk size")
    parser.add_argument("--recreate", action="store_true", help="Recreate Elasticsearch index if exists")
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

    logger.info(f"Starting Stage 6 Elasticsearch Indexing (Index: '{args.index_name}' @ {args.es_url})...")

    builder = ElasticsearchBuilder(es_url=args.es_url, index_name=args.index_name)

    if args.recreate:
        builder.create_index(force_recreate=True)

    ocr_dir = output_dir / "ocr"
    asr_dir = output_dir / "metadata" / "asr"
    if not asr_dir.exists():
        asr_dir = output_dir / "asr"

    ocr_files = list(ocr_dir.glob("*.json")) if ocr_dir.exists() else []
    asr_files = list(asr_dir.glob("*.json*")) if asr_dir.exists() else []

    logger.info(f"Found {len(ocr_files)} OCR JSON file(s) and {len(asr_files)} ASR file(s).")

    documents_map = {}

    # Load OCR records
    for ofile in ocr_files:
        records = builder.load_json_records(ofile)
        for r in records:
            img = str(r.get("image", ""))
            kf_id = Path(img).stem
            if not kf_id:
                continue
            documents_map[kf_id] = {
                "keyframe_id": kf_id,
                "video_id": ofile.stem,
                "image_path": img,
                "ocr_text": str(r.get("text", "")),
                "asr_text": "",
                "caption_text": "",
                "object_text": "",
                "dense_ocr_text": "",
                "merged_text": str(r.get("text", "")),
            }

    # Merge ASR records
    for afile in asr_files:
        records = builder.load_json_records(afile)
        for r in records:
            kf_id = str(r.get("keyframe_id", ""))
            if not kf_id:
                continue
            asr_text = ""
            meta = r.get("asr_metadata", {})
            if isinstance(meta, dict):
                asr_text = str(meta.get("asr_text_6s", ""))

            if kf_id in documents_map:
                documents_map[kf_id]["asr_text"] = asr_text
                documents_map[kf_id]["merged_text"] += f" {asr_text}"
            else:
                documents_map[kf_id] = {
                    "keyframe_id": kf_id,
                    "video_id": str(r.get("video_id", afile.stem.replace("_asr", ""))),
                    "image_path": str(r.get("image_path", "")),
                    "ocr_text": "",
                    "asr_text": asr_text,
                    "caption_text": "",
                    "object_text": "",
                    "dense_ocr_text": "",
                    "merged_text": asr_text,
                }

    if not documents_map:
        logger.warning("No document records found to index into Elasticsearch.")
        sys.exit(0)

    logger.info(f"Indexing {len(documents_map)} combined document(s) into Elasticsearch...")
    count = builder.bulk_import_documents(documents_map.values(), batch_size=args.batch_size)
    logger.info(f"Stage 6 Elasticsearch Indexing completed. Successfully indexed {count} document(s).")


if __name__ == "__main__":
    main()
