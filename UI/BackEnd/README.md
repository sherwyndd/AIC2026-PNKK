# AIC 2026 — FastAPI Search Backend

## Cấu trúc

```
BackEnd/
├── main.py                  ← FastAPI app entry point
├── schemas.py               ← Pydantic request/response models
├── models/
│   ├── __init__.py
│   ├── pe_core_encoder.py   ← PE-Core-L (open_clip, 1024-d, tắt mặc định)
│   └── siglip_encoder.py    ← SigLIP2 (transformers AutoModel, 768-d, bật mặc định)
├── search/
│   ├── __init__.py
│   ├── qdrant_search.py     ← Qdrant client + query helpers
│   └── rrf.py               ← Reciprocal Rank Fusion
└── tests/                   ← Các script test/debug, diagnostic
```

## Khởi động

```bash
conda activate aic2026_backend
cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd

# Chạy trong tmux để giữ session (khuyến nghị)
tmux new -s backend
uvicorn main:app --host 0.0.0.0 --port 8888

# Hoặc với reload (dev mode)
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

## Endpoints

### `GET /health`
Kiểm tra server, model đã load xong chưa, Qdrant còn kết nối không.

```bash
curl http://localhost:8888/health
```

```json
{"status":"ok","models_loaded":true,"qdrant_connected":true}
```

### `POST /api/search`

```bash
curl -X POST http://localhost:8888/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "a man riding a red motorbike",
    "top_k": 20,
    "rrf_k": 60,
    "video_list": []
  }'
```

**Response:**
```json
{
  "query": "a man riding a red motorbike",
  "top_k": 20,
  "rrf_k": 60,
  "results": [
    {
      "rank": 1,
      "score": 0.032787,
      "video_id": "L24_V005",
      "keyframe_id": "L24_V005_kf_0187",
      "frame_idx": 187,
      "image_url": "/images/L24_V005/L24_V005_kf_0187.jpg"
    }
  ]
}
```

## Lưu ý

- Model load **1 lần duy nhất** lúc khởi động (lifespan). Gọi `/health` để biết khi nào xong.
- GPU mặc định: `CUDA_VISIBLE_DEVICES` được set `"6"` ở đầu file main.py (trước khi import torch). Model run internal trên `cuda:0`.
- Hiện tại **chỉ dùng SigLIP2** (flag `USE_PECORE = False` trong main.py). PE-Core-L vẫn có trong code nếu sau này cần bật lại.
- Static files ảnh: `GET /images/<video_id>/<keyframe_id>.jpg`
- CORS `allow_origins=["*"]` — frontend React gọi thẳng được.
- `use_ocr` / `use_asr` trong request body được nhận nhưng **chưa xử lý** (placeholder).
