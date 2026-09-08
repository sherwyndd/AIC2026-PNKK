# Spec: FastAPI Backend cho Text-to-Image Search (PE-Core + SigLIP + RRF)

## 1. Mục tiêu

Xây dựng một FastAPI backend nhận **text query** từ người dùng, encode text bằng **1-2 model độc lập** (PE-Core-L và/hoặc SigLIP2), query song song lên **Qdrant collection** tương ứng, sau đó **hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF)** để trả về danh sách Top-K ảnh cuối cùng cho frontend React.

Hiện tại **chỉ SigLIP2 được bật mặc định** (`USE_PECORE = False` trong main.py) để tối ưu memory và tốc độ. PE-Core-L vẫn sẵn sàng trong code nếu sau này cần bật lại RRF 2 nguồn.

---

## 2. Kiến trúc tổng quan

```
React Frontend
        │  HTTP (CORS: allow_origins=["*"])
        ▼
FastAPI Backend (port 8888, host 0.0.0.0)
   ├── Model PE-Core-L       (load 1 lần lúc startup, GPU cuda:0 — tắt mặc định)
   ├── Model SigLIP2         (load 1 lần lúc startup, GPU cuda:0 — bật mặc định)
   ├── Qdrant Client         (localhost:6333, timeout 120s)
   │      ├── collection: image_pecore   (dim=1024, cosine, unnamed vector)
   │      └── collection: image_siglip   (dim=768,  cosine, unnamed vector)
   ├── RRF Fusion Module
   └── StaticFiles mount     (/images → /AIClub_NAS/core_baotg/nhan/dataset/keyframes)
```

Không dùng Docker. Chạy trực tiếp bằng `uvicorn main:app --host 0.0.0.0 --port 8888` (trong tmux/screen), trong conda env `aic2026_backend`.

---

## 3. Models & Embedding

### 3.1. PE-Core-L (tắt mặc định)
- Model: `hf-hub:timm/PE-Core-L-14-336`
- Thư viện: `open_clip`
- Load:
  ```python
  import open_clip
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:timm/PE-Core-L-14-336"
  )
  tokenizer = open_clip.get_tokenizer("hf-hub:timm/PE-Core-L-14-336")
  ```
- Encode text: `model.encode_text(tokens, normalize=True)`
- Embedding dim: **1024** (float32), L2 normalized
- Flag bật/tắt: `USE_PECORE = True/False` ở đầu `main.py`

### 3.2. SigLIP2 (bật mặc định)
- Model: `google/siglip2-base-patch16-224`
- Thư viện: `transformers` (`AutoModel`, `AutoProcessor`)
- **Lưu ý kỹ thuật quan trọng**: SiglipModel.forward() cần BOTH text + image inputs. Để lấy được `text_embeds` đã qua projection head (không gian contrastive đúng), ta truyền một dummy `pixel_values` zero-tensor cùng với text thật:
  ```python
  from transformers import AutoModel, AutoProcessor
  siglip_model = AutoModel.from_pretrained(
      "google/siglip2-base-patch16-224", dtype=torch.float16
  ).to("cuda:0").eval()
  siglip_processor = AutoProcessor.from_pretrained(
      "google/siglip2-base-patch16-224"
  )

  text_inputs = siglip_processor(text=[query], padding="max_length",
                                 max_length=64, truncation=True, return_tensors="pt")
  input_ids = text_inputs["input_ids"].to("cuda:0")
  B = input_ids.shape[0]
  dummy_pixel = torch.zeros(B, 3, 224, 224, dtype=torch.float16, device="cuda:0")

  with torch.inference_mode():
      outputs = siglip_model(input_ids=input_ids, pixel_values=dummy_pixel)
  text_emb = outputs["text_embeds"][0].cpu().float().numpy()  # shape (768,)
  ```
- Embedding dim: **768** (float32), L2 normalized BỞI SigLipModel.forward() luôn.
- Dtype trên CUDA: `float16`, cast sang `float32` trước khi trả về numpy cho Qdrant.

### 3.3. Lifecycle
- Cả 2 model (1 nếu PE-Core tắt) **load một lần duy nhất** khi FastAPI khởi động, dùng `lifespan` context manager (không load lại trong mỗi request).
- GPU visibility cố định bằng `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")` đặt ở ĐẦU file main.py, TRƯỚC KHI import torch/transformers. Sau đó model được `.to("cuda:0")` (device logic 0 của GPU visible).

---

## 4. Qdrant

### 4.1. Kết nối
- Host: `127.0.0.1` (localhost), Port: `6333`
- Client timeout: `120s` (cả client init lẫn per-query timeout)
- Kết nối được tạo **lazy 1 lần** bởi `get_qdrant_client()` singleton.

### 4.2. Collections

| Thuộc tính      | image_pecore                 | image_siglip                       |
|-----------------|------------------------------|-------------------------------------|
| Vector field    | unnamed (default)            | unnamed (default)                   |
| Dimension       | 1024                         | 768                                 |
| Distance metric | Cosine                       | Cosine                              |
| Model gốc       | timm/PE-Core-L-14-336        | google/siglip2-base-patch16-224     |
| Trạng thái sử dụng | tắt mặc định              | bật mặc định                        |

### 4.3. Payload (giống nhau ở cả 2 collection)
```json
{
  "video_id":    "L24_V005",
  "keyframe_id": "L24_V005_kf_0187",
  "frame_idx":   187,
  "image_path":  "L24_V005/L24_V005_kf_0187.jpg"
}
```
`image_path` là đường dẫn tương đối, root vật lý: `/AIClub_NAS/core_baotg/nhan/dataset/keyframes/`.

### 4.4. Query mẫu
```python
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

client = QdrantClient("127.0.0.1", port=6333, timeout=120)
hits = client.query_points(
    collection_name="image_siglip",
    query=text_emb.tolist(),
    limit=fetch_depth,
    query_filter=Filter(
        must=[FieldCondition(key="video_id", match=MatchAny(any=["L01_V001", ...]))]
    ) if video_list else None,
    timeout=120,
).points
```
Không cần truyền `using=` vì vector là unnamed.

### 4.5. Filter theo video (tùy chọn, nâng cao)
- Mặc định: **không khoanh vùng** — tìm trên toàn bộ database.
- Nếu client truyền `video_list` (không rỗng), áp dụng `MatchAny` trên field `video_id`.

---

## 5. RRF (Reciprocal Rank Fusion)

### 5.1. Công thức
Với mỗi ảnh xuất hiện ở rank `r` (1-indexed) trong danh sách kết quả của một model:

```
score(image) = Σ over models   1 / (k + rank_in_model)
```

Ảnh không xuất hiện trong kết quả của 1 model nào đó thì đơn giản không cộng phần điểm từ model đó (không phạt điểm âm). **Score được normalize** bằng cách chia cho `max_score = (num_active_models) * (1/(k+1))` và làm tròn 6 chữ số thập phân, để score cuối cùng rơi vào khoảng (0, 1] trực quan hơn.

### 5.2. Tham số
- **`k`**: Truyền từ frontend qua field `rrf_k` của request, mặc định backend `60` nếu không có. Đây là giá trị chuẩn phổ biến.
- **`fetch_depth`** (số lượng lấy từ mỗi model *trước khi* fuse): `fetch_depth = max(100, top_k * 5)` — đảm bảo đủ overlap cho RRF có ý nghĩa.
- **`top_k`** (số ảnh trả về cuối cùng sau fuse): lấy từ input người dùng (React settings), bắt buộc có trong request.

### 5.3. Thuật toán (giả mã)
```python
def rrf_fuse(results_by_model: dict[str, list[Hit]], k: int, top_k: int) -> list[dict]:
    scores: dict[str, float] = {}    # key = keyframe_id
    meta:   dict[str, dict] = {}     # payload cache

    for _model_name, hits in results_by_model.items():
        for rank, hit in enumerate(hits, start=1):
            key = hit.payload["keyframe_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in meta:
                meta[key] = hit.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    non_empty_models = sum(1 for h in results_by_model.values() if h)
    max_score = non_empty_models * (1.0 / (k + 1)) if non_empty_models else 1.0

    return [
        {"score": round((score / max_score) if max_score > 0 else 0.0, 6), **meta[key]}
        for key, score in ranked
    ]
```

Chạy 2 query Qdrant **song song** (dùng `asyncio.gather` + `run_in_executor` với `ThreadPoolExecutor(max_workers=2)`) để giảm latency.

---

## 6. API Endpoints

### 6.1. `POST /api/search`

**Request body:**
```json
{
  "query": "a man riding a red motorbike",
  "top_k": 20,
  "rrf_k": 60,
  "video_list": [],
  "use_ocr": false,
  "use_asr": false
}
```

| Field        | Type          | Bắt buộc | Ghi chú |
|--------------|---------------|----------|---------|
| `query`      | string        | Có       | Text mô tả cần tìm (không rỗng, auto strip whitespace) |
| `top_k`      | int           | Có       | Số ảnh trả về sau fusion, > 0 |
| `rrf_k`      | int           | Không    | Hệ số k của RRF, default backend = 60 |
| `video_list` | list[string]  | Không    | Nếu rỗng/None → tìm toàn bộ DB. Nếu có → lọc theo `video_id` |
| `use_ocr`    | bool          | Không    | **Chưa implement** — nhận field nhưng backend ignore |
| `use_asr`    | bool          | Không    | **Chưa implement**, tương tự `use_ocr` |

**Response body:**
```json
{
  "query": "a man riding a red motorbike",
  "top_k": 20,
  "rrf_k": 60,
  "results": [
    {
      "rank": 1,
      "score": 0.0328,
      "video_id": "L24_V005",
      "keyframe_id": "L24_V005_kf_0187",
      "frame_idx": 187,
      "image_url": "/images/L24_V005/L24_V005_kf_0187.jpg"
    }
  ]
}
```

`image_url` là **root-relative URL** (`/images/...`) để hoạt động đúng cả truy cập trực tiếp lẫn SSH tunnel.

### 6.2. `GET /health`
**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "qdrant_connected": true
}
```
Dùng để frontend/monitoring kiểm tra server còn sống và model đã load xong chưa.

### 6.3. `GET /`
Serve tệp `index.html` của Frontend (nếu có) — dùng để truy cập trực tiếp Web UI qua port backend.

### 6.4. `GET /images/{video_id}/{keyframe_filename}`
Static file serve cho keyframe image (JPG).

---

## 7. Serve ảnh tĩnh

FastAPI mount thư mục ảnh gốc thành route web:

```python
from fastapi.staticfiles import StaticFiles

KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
app.mount("/images", StaticFiles(directory=KEYFRAME_ROOT), name="images")
```

Backend build `image_url` = `/images/` + `image_path` (path tương đối từ payload Qdrant). Frontend dùng thẳng trong `<img src="...">`.

---

## 8. CORS

Bắt buộc bật (vì React Vite dev server và FastAPI khác origin):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 9. Auth / Rate Limit

**Không cần.** Nhóm nội bộ test/thi đấu, bỏ qua để giảm độ trễ.

---

## 10. Cấu trúc thư mục BackEnd

```
BackEnd/
├── main.py                # FastAPI app: lifespan, routes, CORS, static mount
├── schemas.py             # Pydantic models (SearchRequest, SearchResponse, ImageResult, HealthResponse)
├── requirements.txt
├── requirements_new.txt
├── README.md              # Hướng dẫn khởi động ngắn gọn
├── spec.md                # Spec này (chi tiết)
├── models/
│   ├── __init__.py
│   ├── pe_core_encoder.py # load_pecore(), encode_text_pecore() — tắt mặc định
│   └── siglip_encoder.py  # load_siglip(), encode_text_siglip(), encode_images_siglip()
├── search/
│   ├── __init__.py
│   ├── qdrant_search.py   # get_qdrant_client(), is_qdrant_healthy(), query_collection()
│   └── rrf.py             # rrf_fuse()
└── tests/                 # Script test/debug/diagnostic (không chạy trong production)
    ├── run_sanity_check.py
    ├── test_query_sanity.py
    ├── test_siglip2_diagnostic.py
    ├── check_qdrant_collection.py
    ├── test_encoding_speed.py
    ├── test_reindex_sample.py
    ├── test_various_queries.py
    ├── test_embedding_compatibility.py
    ├── test_siglip_only.py
    ├── test_latency.py
    ├── fast_import_npy_to_qdrant.py
    └── reindex_siglip_collection.py
```

---

## 11. Luồng xử lý 1 request `/api/search`

1. **Validate request** Pydantic: `query` không rỗng (auto strip), `top_k > 0`, `rrf_k >= 1` nếu có.
2. **Kiểm tra models đã sẵn sàng** chưa (flag `_models_loaded`). Nếu chưa → HTTP 503 "Models are still loading".
3. **Step 1 — Encode text song song** (`_encode_both`):
   - Nếu `USE_PECORE=True`: chạy `encode_text_pecore` và `encode_text_siglip` song song qua thread pool.
   - Nếu `USE_PECORE=False`: chỉ chạy `encode_text_siglip` qua thread pool (giữ `pecore_emb=None`).
4. **Step 2 — Query Qdrant song song** (`_query_both`):
   - Query `image_siglip` collection (luôn).
   - Query `image_pecore` collection (nếu `USE_PECORE=True`).
   - Dùng `ThreadPoolExecutor(max_workers=2)`.
   - `fetch_depth = max(100, top_k * 5)`.
   - Filter `video_list` (nếu có) qua `MatchAny` field `video_id`.
5. **Step 3 — RRF Fusion** (`rrf_fuse`):
   - Gộp 2 (hoặc 1) ranked list, dedupe theo `keyframe_id`, cộng RRF score.
   - Sort desc, cắt `top_k` phần tử đầu.
   - Normalize score về (0,1] và làm tròn 6 dp.
6. **Step 4 — Build response**:
   - Thêm `rank` (1-indexed).
   - Build `image_url` root-relative `/images/...`.
7. **Return** `SearchResponse` JSON.

---

## 12. Việc KHÔNG nằm trong scope hiện tại

- Không xử lý OCR / ASR / Object text search (để dành phase sau).
- Không có endpoint upload/index ảnh mới (dữ liệu AIC cố định, đã vector hóa sẵn).
- Không Docker hóa.
- Không auth/API key.
- Không rate limiting.
