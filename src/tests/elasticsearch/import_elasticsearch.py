from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator, TypedDict

from elasticsearch import Elasticsearch
from elasticsearch import helpers
from tqdm import tqdm


ELASTICSEARCH_URL = "http://127.0.0.1:9200"
INDEX_NAME = "aic2026_elastics_text"
BATCH_SIZE = 1000

OCR_DIR = Path("/AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/ocr_final")
ASR_DIR = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr")
CAPTIONS_DIR = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata/captions")
OBJECTS_DIR = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata/objects")
KEYFRAMES_MARKER = "keyframes"

TEST_VIDEO = None


class OcrDocument(TypedDict):
    image_path: str
    keyframe_id: str
    video_id: str
    frame_idx: int
    ocr_text: str


class AsrDocument(TypedDict):
    image_path: str
    keyframe_id: str
    video_id: str
    frame_idx: int
    asr_text: str


class CaptionDocument(TypedDict):
    keyframe_id: str
    caption_text: str


class ObjectDocument(TypedDict):
    keyframe_id: str
    object_text: str


class KeyframeDocument(TypedDict):
    image_path: str
    keyframe_id: str
    video_id: str
    frame_idx: int
    ocr_text: str
    asr_text: str
    caption_text: str
    object_text: str
    dense_ocr_text: str
    merged_text: str


def create_index(client: Elasticsearch, index_name: str = INDEX_NAME) -> None:
    if client.indices.exists(index=index_name):
        return
    client.indices.create(index=index_name, body=index_mapping())


def index_mapping() -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "image_path": {"type": "keyword"},
                "keyframe_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "frame_idx": {"type": "integer"},
                "ocr_text": {"type": "text"},
                "asr_text": {"type": "text"},
                "caption_text": {"type": "text"},
                "object_text": {"type": "text"},
                "dense_ocr_text": {"type": "text"},
                "merged_text": {"type": "text"},
            }
        }
    }


def load_ocr(ocr_dir: Path = OCR_DIR) -> dict[str, OcrDocument]:
    """Keyed by keyframe_id (not image_path) for robust merging."""
    documents: dict[str, OcrDocument] = {}
    json_files = list_metadata_files(ocr_dir)
    print(f"\n[OCR] dir: {ocr_dir.resolve()}")
    print(f"[OCR] found {len(json_files)} files")
    if TEST_VIDEO is not None:
        json_files = [ocr_dir / f"{TEST_VIDEO}.json"]

    total_records = 0
    skipped_empty_path = 0
    for json_path in tqdm(json_files, desc="Loading OCR"):
        records = read_json_records(json_path)
        total_records += len(records)
        for record in records:
            raw_image = str(record.get("image", ""))
            image_path = normalize_image_path(raw_image)
            if not image_path:
                skipped_empty_path += 1
                continue
            keyframe_id = keyframe_id_from_image_path(image_path)
            documents[keyframe_id] = {
                "image_path": image_path,
                "keyframe_id": keyframe_id,
                "video_id": video_id_from_image_path(image_path),
                "frame_idx": frame_idx_from_image_path(image_path),
                "ocr_text": as_text(record.get("text_hybrid", "")),
            }

    print(f"[OCR] total records read: {total_records}")
    print(f"[OCR] skipped (empty image path): {skipped_empty_path}")
    print(f"[OCR] unique keyframes loaded: {len(documents)}")
    if documents:
        sample_key = next(iter(documents))
        print(f"[OCR] sample: {sample_key} -> {documents[sample_key]}")
    return documents


def load_asr(asr_dir: Path = ASR_DIR) -> dict[str, AsrDocument]:
    documents: dict[str, AsrDocument] = {}
    json_files = sorted(asr_dir.glob("*.jsonl"))
    print(f"\n[ASR] dir: {asr_dir.resolve()}")
    print(f"[ASR] found {len(json_files)} files")
    if TEST_VIDEO is not None:
        json_files = [asr_dir / f"{TEST_VIDEO}_asr.jsonl"]

    total_records = 0
    skipped_empty_path = 0
    for json_path in tqdm(json_files, desc="Loading ASR"):
        records = read_json_records(json_path)
        total_records += len(records)
        for record in records:
            raw_image = str(record.get("image_path", ""))
            image_path = normalize_image_path(raw_image)
            if not image_path:
                skipped_empty_path += 1
                continue

            asr_metadata = record.get("asr_metadata", {})
            if not isinstance(asr_metadata, dict):
                asr_metadata = {}

            keyframe_id = as_text(record.get("keyframe_id", "")) or keyframe_id_from_image_path(image_path)

            try:
                frame_idx = int(record.get("frame_idx", frame_idx_from_image_path(image_path)))
            except (TypeError, ValueError):
                frame_idx = frame_idx_from_image_path(image_path)

            documents[keyframe_id] = {
                "image_path": image_path,
                "keyframe_id": keyframe_id,
                "video_id": as_text(record.get("video_id", "")) or video_id_from_image_path(image_path),
                "frame_idx": frame_idx,
                "asr_text": as_text(asr_metadata.get("asr_text_6s", "")),
            }

    print(f"[ASR] total records read: {total_records}")
    print(f"[ASR] skipped (empty image path): {skipped_empty_path}")
    print(f"[ASR] unique keyframes loaded: {len(documents)}")
    if documents:
        sample_key = next(iter(documents))
        print(f"[ASR] sample: {sample_key} -> {documents[sample_key]}")
    return documents


def load_captions(captions_dir: Path = CAPTIONS_DIR) -> dict[str, CaptionDocument]:
    documents: dict[str, CaptionDocument] = {}
    files = list_metadata_files(captions_dir)
    print(f"\n[CAPTIONS] dir: {captions_dir.resolve()}")
    print(f"[CAPTIONS] found {len(files)} files")
    for json_path in tqdm(files, desc="Loading captions"):
        for record in read_json_records(json_path):
            keyframe_id = as_text(record.get("keyframe_id", ""))
            if keyframe_id:
                documents[keyframe_id] = {
                    "keyframe_id": keyframe_id,
                    "caption_text": as_text(record.get("caption_en", "")),
                }
    print(f"[CAPTIONS] unique keyframes loaded: {len(documents)}")
    return documents


def load_objects(objects_dir: Path = OBJECTS_DIR) -> dict[str, ObjectDocument]:
    documents: dict[str, ObjectDocument] = {}
    files = list_metadata_files(objects_dir)
    print(f"\n[OBJECTS] dir: {objects_dir.resolve()}")
    print(f"[OBJECTS] found {len(files)} files")
    for json_path in tqdm(files, desc="Loading objects"):
        for record in read_json_records(json_path):
            keyframe_id = as_text(record.get("keyframe_id", ""))
            if keyframe_id:
                object_text = as_text(record.get("detected_objects", ""))
                if not object_text and isinstance(record.get("unique_objects"), list):
                    object_text = " ".join(as_text(item) for item in record["unique_objects"])
                documents[keyframe_id] = {
                    "keyframe_id": keyframe_id,
                    "object_text": object_text,
                }
    print(f"[OBJECTS] unique keyframes loaded: {len(documents)}")
    return documents


def merge_documents(
    ocr_documents: dict[str, OcrDocument],
    asr_documents: dict[str, AsrDocument],
    caption_documents: dict[str, CaptionDocument] | None = None,
    object_documents: dict[str, ObjectDocument] | None = None,
) -> dict[str, KeyframeDocument]:
    """Merge is keyed by keyframe_id everywhere -> O(1) lookups, no fragile path matching."""
    caption_documents = caption_documents or {}
    object_documents = object_documents or {}

    all_keyframe_ids = set(ocr_documents) | set(asr_documents) | set(caption_documents) | set(object_documents)
    print(f"\n[MERGE] union of keyframe_ids across all 4 sources: {len(all_keyframe_ids)}")

    merged: dict[str, KeyframeDocument] = {}
    for keyframe_id in sorted(all_keyframe_ids):
        ocr_document = ocr_documents.get(keyframe_id)
        asr_document = asr_documents.get(keyframe_id)
        caption_document = caption_documents.get(keyframe_id)
        object_document = object_documents.get(keyframe_id)

        image_path = (
            (ocr_document or {}).get("image_path")
            or (asr_document or {}).get("image_path")
            or keyframe_id
        )
        video_id = (
            (ocr_document or {}).get("video_id")
            or (asr_document or {}).get("video_id")
            or video_id_from_image_path(keyframe_id)
        )
        frame_idx = (
            (ocr_document or {}).get("frame_idx")
            if ocr_document is not None
            else (asr_document or {}).get("frame_idx", frame_idx_from_image_path(keyframe_id))
        )

        ocr_text = (ocr_document or {}).get("ocr_text", "")
        asr_text = (asr_document or {}).get("asr_text", "")
        caption_text = (caption_document or {}).get("caption_text", "")
        object_text = (object_document or {}).get("object_text", "")
        dense_ocr_text = ""

        merged[keyframe_id] = {
            "image_path": image_path,
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_idx": frame_idx,
            "ocr_text": ocr_text,
            "asr_text": asr_text,
            "caption_text": caption_text,
            "object_text": object_text,
            "dense_ocr_text": dense_ocr_text,
            "merged_text": merge_text_fields(ocr_text, asr_text, caption_text, object_text, dense_ocr_text),
        }

    print(f"[MERGE] total merged documents: {len(merged)}")
    return merged


def bulk_insert(
    client: Elasticsearch,
    documents: dict[str, KeyframeDocument],
    index_name: str = INDEX_NAME,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, int]:
    items = list(documents.items())
    print(f"\n[BULK] preparing to insert {len(items)} documents into '{index_name}'")
    if not items:
        print("[BULK] nothing to insert (documents dict is empty) - skipping")
        return 0, 0

    total_success = 0
    total_errors = 0
    for start in tqdm(range(0, len(items), batch_size), desc="Bulk inserting"):
        batch = items[start: start + batch_size]
        success, errors = helpers.bulk(
            client,
            build_actions(batch, index_name),
            raise_on_error=False,
            stats_only=False,
        )
        total_success += success
        if errors:
            total_errors += len(errors)
            print(f"[BULK] batch at offset {start}: {len(errors)} errors, first error:")
            print(json.dumps(errors[0], ensure_ascii=False, indent=2, default=str))

    print(f"[BULK] total success: {total_success}, total errors: {total_errors}")
    return total_success, total_errors


def build_actions(
    documents: Iterable[tuple[str, KeyframeDocument]],
    index_name: str,
) -> Iterator[dict[str, Any]]:
    for keyframe_id, document in documents:
        yield {
            "_index": index_name,
            "_id": keyframe_id,
            "_source": document,
        }


def list_metadata_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted((*directory.glob("*.json"), *directory.glob("*.jsonl")))


def read_json_records(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        if json_path.suffix.lower() == ".jsonl":
            records = []
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {json_path}:{line_number}") from exc
                if isinstance(item, dict):
                    records.append(item)
            return records
        data = json.load(file)

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON object or list in {json_path}")

    records: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            records.append(item)
    return records


def normalize_image_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    if KEYFRAMES_MARKER in parts:
        marker_index = parts.index(KEYFRAMES_MARKER)
        return "/".join(parts[marker_index:])
    return normalized.lstrip("/")


def frame_idx_from_image_path(image_path: str) -> int:
    stem = Path(image_path).stem
    if "_kf_" in stem:
        try:
            return int(stem.split("_kf_")[1])
        except ValueError:
            return -1
    return -1


def video_id_from_image_path(image_path: str) -> str:
    parts = Path(image_path).parts
    if len(parts) >= 2 and parts[0] == KEYFRAMES_MARKER:
        return parts[1]
    filename = Path(image_path).stem
    if "_kf_" in filename:
        return filename.split("_kf_", maxsplit=1)[0]
    return ""


def keyframe_id_from_image_path(image_path: str) -> str:
    return Path(image_path).stem


def merge_text_fields(*fields: str) -> str:
    return " ".join(field for field in fields if field)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def main() -> None:
    client = Elasticsearch([ELASTICSEARCH_URL])
    print("[ES] info:", client.info().get("version", {}))

    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"[ES] deleted existing index '{INDEX_NAME}'")
    create_index(client)
    print(f"[ES] index '{INDEX_NAME}' ready")

    ocr_documents = load_ocr()
    asr_documents = load_asr()
    caption_documents = load_captions()
    object_documents = load_objects()

    if not (ocr_documents or asr_documents or caption_documents or object_documents):
        print("\n[FATAL] Tất cả 4 nguồn đều rỗng! Kiểm tra lại đường dẫn OCR_DIR/ASR_DIR/CAPTIONS_DIR/OBJECTS_DIR.")
        sys.exit(1)

    merged_documents = merge_documents(ocr_documents, asr_documents, caption_documents, object_documents)

    if not merged_documents:
        print("\n[FATAL] merged_documents rỗng dù các nguồn có dữ liệu - có bug trong merge_documents.")
        sys.exit(1)

    from itertools import islice
    import pprint

    print("\nSample merged documents:\n")
    for _, doc in islice(merged_documents.items(), 2):
        pprint.pp(doc)

    success, errors = bulk_insert(client, merged_documents)
    client.indices.refresh(index=INDEX_NAME)

    count_resp = client.count(index=INDEX_NAME)
    print(f"\n[ES] count() after refresh: {count_resp.get('count')}")

    print(f"\nTotal OCR: {len(ocr_documents)}")
    print(f"Total ASR: {len(asr_documents)}")
    print(f"Total captions: {len(caption_documents)}")
    print(f"Total objects: {len(object_documents)}")
    print(f"Total merged documents: {len(merged_documents)}")
    print(f"Bulk success: {success}, bulk errors: {errors}")
    print("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[UNCAUGHT EXCEPTION]")
        traceback.print_exc()
        sys.exit(1)