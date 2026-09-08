import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm

_logger = logging.getLogger(__name__)


def get_default_es_mapping() -> Dict[str, Any]:
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


class ElasticsearchBuilder:
    """Core Elasticsearch Indexing Builder for keyframe text metadata."""

    def __init__(self, es_url: str = "http://127.0.0.1:9200", index_name: str = "aic2026_elastics_text"):
        self.es_url = es_url
        self.index_name = index_name
        self.client = Elasticsearch(self.es_url)

    def create_index(self, force_recreate: bool = False):
        if force_recreate and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            _logger.info(f"Deleted existing index: {self.index_name}")

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=get_default_es_mapping())
            _logger.info(f"Created index: {self.index_name}")

    def bulk_import_documents(self, documents: Iterable[Dict[str, Any]], batch_size: int = 1000) -> int:
        def generate_actions():
            for doc in documents:
                keyframe_id = doc.get("keyframe_id", "")
                yield {
                    "_index": self.index_name,
                    "_id": keyframe_id if keyframe_id else None,
                    "_source": doc,
                }

        self.create_index(force_recreate=False)
        success, _ = bulk(self.client, generate_actions(), chunk_size=batch_size, stats_only=True)
        self.client.indices.refresh(index=self.index_name)
        return success

    @staticmethod
    def load_json_records(file_path: Path) -> List[Dict[str, Any]]:
        if not file_path.exists():
            return []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    return [json.loads(line) for line in f if line.strip()]
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except Exception as exc:
            _logger.error(f"Error loading {file_path}: {exc}")
            return []
