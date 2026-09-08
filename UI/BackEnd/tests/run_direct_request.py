import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import json
from search.temporal_search import StageQuery, TemporalSearchRequest, solve_temporal_search

req = TemporalSearchRequest(
    stages=[
        StageQuery(
            stage_index=0,
            text_query="red car driving along highway",
            ocr_query=None,
            asr_query=None,
            weight_text=1.0,
            weight_ocr=1.0,
            weight_asr=1.0
        ),
        StageQuery(
            stage_index=1,
            text_query="driver parking vehicle near building",
            ocr_query="PARKING",
            asr_query=None,
            weight_text=1.0,
            weight_ocr=1.5,
            weight_asr=1.0
        )
    ],
    global_context="urban traffic documentary",
    context_bonus_weight=0.15,
    max_gap_seconds=120.0,
    top_k_per_stage=100,
    top_k_videos=3
)

resp = solve_temporal_search(req)
print("STATUS_CODE: 200")
print("RESPONSE_JSON:")
print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
