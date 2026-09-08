import os
import sys

# KHÔNG set CUDA_VISIBLE_DEVICES ở đây — đọc từ shell env var (CUDA_VISIBLE_DEVICES=X python ...)
# Nếu chưa set, để main.py tự set default ("6").

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

payload = {
    "stages": [
        {
            "stage_index": 0,
            "text_query": "red car driving along highway",
            "ocr_query": None,
            "asr_query": None,
            "weight_text": 1.0,
            "weight_ocr": 1.0,
            "weight_asr": 1.0
        },
        {
            "stage_index": 1,
            "text_query": "driver parking vehicle near building",
            "ocr_query": "PARKING",
            "asr_query": None,
            "weight_text": 1.0,
            "weight_ocr": 1.5,
            "weight_asr": 1.0
        }
    ],
    "global_context": "urban traffic documentary",
    "context_bonus_weight": 0.15,
    "max_gap_seconds": 120.0,
    "top_k_per_stage": 100,
    "top_k_videos": 3
}

response = client.post("/api/v1/temporal-search/solve", json=payload)
print(f"STATUS_CODE: {response.status_code}")
print("RESPONSE_JSON:")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
