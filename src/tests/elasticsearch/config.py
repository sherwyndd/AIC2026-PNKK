from pathlib import Path
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ELASTIC_URL = "http://127.0.0.1:9200"
INDEX_NAME = "aic2026_elastics_text"

OCR_DIR = Path("outputs/ocr")

BATCH_SIZE = 1000
def parse_video_id(json_path: Path):
    return json_path.stem
def convert_doc(video_id, item):
    image_path = Path(item["image"])

    return {
        "_index": INDEX_NAME,
        "_id": f"{video_id}_{item['doc_id']}",
        "_source": {
            "doc_id": f"{video_id}_{item['doc_id']}",
            "video_id": video_id,
            "frame_idx": item["doc_id"] + 1,
            "image_path": str(image_path.relative_to(KEYFRAME_ROOT)),
            "source": "ocr",
            "text": item["text"],
        },
    }
