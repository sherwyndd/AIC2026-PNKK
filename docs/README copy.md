# K2PN — Hệ thống Tìm kiếm Video/Ảnh (AI Challenge 2026)
## Tổng quan pipeline

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                        VIDEO PREPROCESSING PIPELINE (ROOT PROJECT)          │
  │ ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                  │
  │ │ 01_shot_cut    │→│02_keyframe_ext │→│03_ASR / 05_OBJ  │→ ...              │
  │ │ (TransNetV2)   │  │ (every-8th +   │  │ (Whisper +     │                  │
  │ │                │  │  CLIP L2 filter)│  │  YOLOv8)       │                  │
  │ └────────┬───────┘  └────────┬───────┘  └────────┬───────┘                  │
  │          │ Keyframes (JPG)  │ Metadata (JSON)    │                          │
  └──────────┼──────────────────┼────────────────────┼──────────────────────────┘
             │                  │                    │
             ▼                  ▼                    │
  ┌─────────────────────────────────────────┐       │
  │          OFFLINE EMBEDDING & INDEX       │       │
  │ ┌─────────────────────────────────────┐  │       │
  │ │ Encode all keyframes bằng SigLIP2   │  │       │
  │ │ (PE-Core-L option)                  │  │       │
  │ │ → Lưu .npy feature files            │  │       │
  │ └────────────────────┬────────────────┘  │       │
  │                      ▼                   │       │
  │ ┌─────────────────────────────────────┐  │       │
  │ │ fast_import_npy_to_qdrant.py:       │  │       │
  │ │ Upload vectors & payloads lên       │  │       │
  │ │ Qdrant collections                  │  │       │
  │ │   • image_siglip  (dim=768, cosine) │  │       │
  │ │   • image_pecore  (dim=1024,cosine) │  │       │
  │ └────────────────────┬────────────────┘  │       │
  └──────────────────────┼───────────────────┘       │
                         │                            │
  ┌──────────────────────┼────────────────────────────┘─────────────────────────┐
  │      UI LAYER (thư mục này: /UI)                                             │
  │                                                                               │
  │   ┌───────────────────────────┐     HTTP/JSON POST :8888     ┌──────────────┐│
  │   │   FrontEnd (React + Vite) │ ──── /api/search ──────────▶│   BackEnd    ││
  │   │                           │ ◀──── results JSON ───────── │  (FastAPI)   ││
  │   │ • Header (TopK, Reset)    │                              │              ││
  │   │ • Sidebar (Search Panels) │◀── health/JSON ──────────── │ • Lifespan:  ││
  │   │   - Text/ASR/OCR/Object   │   (check backend live)      │   load model ││
  │   │ • ResultGrid (responsive)│                              │ • SigLIP2    ││
  │   │ • ResultDetailModal      │   GET :8888/images/...       │ • PE-Core-L  ││
  │   └──────────────┬────────────┘ ◀────── keyframe JPGs ───── │ (opt, OFF)  ││
  │                  │ (nếu dùng index.html standalone)         │              ││
  │                  └──────── GET :8888/ ─────────────────────▶│ • Qdrant     ││
  │                     serve index.html + /images/             │   query      ││
  │                                                             │ • RRF fuse   ││
  │   Test_UI/          BackEnd/                                │ + StaticFile ││
  │   React Vite proj   FastAPI Python                          │   (/images)  ││
  └─────────────────────────────────────────────────────────────┴──────────────┘│
```

---

## Cấu trúc thư mục UI

```
UI/
├── README.md                     ← File này — tài liệu tổng thể
├── doc.txt                       ← Shortcut command cũ
├── BackEnd/
│   ├── main.py                   ← FastAPI app entry point (:8888)
│   ├── schemas.py                ← Pydantic request/response models
│   ├── requirements.txt
│   ├── requirements_new.txt
│   ├── README.md                 ← Hướng dẫn khởi động ngắn gọn
│   ├── spec.md                   ← Spec chi tiết backend kỹ thuật
│   ├── models/
│   │   ├── __init__.py
│   │   ├── siglip_encoder.py     ← Model google/siglip2-base-patch16-224 (bật)
│   │   └── pe_core_encoder.py    ← Model timm/PE-Core-L-14-336 (tắt mặc định)
│   ├── search/
│   │   ├── __init__.py
│   │   ├── qdrant_search.py      ← Qdrant client + query API
│   │   └── rrf.py                ← Reciprocal Rank Fusion
│   └── tests/                    ← Script test/debug/diagnostic (12 file)
│       ├── run_sanity_check.py
│       ├── test_query_sanity.py
│       ├── test_siglip2_diagnostic.py
│       ├── check_qdrant_collection.py
│       ├── test_encoding_speed.py
│       ├── test_reindex_sample.py
│       ├── test_various_queries.py
│       ├── test_embedding_compatibility.py
│       ├── test_siglip_only.py
│       ├── test_latency.py
│       ├── fast_import_npy_to_qdrant.py
│       └── reindex_siglip_collection.py
└── Test_UI/                      ← FrontEnd React + Vite project
    ├── index.html                ← [OLD CDN STANDALONE] serve trực tiếp qua /
    ├── package.json              ← Dependencies (React 18, Vite 5, Tailwind, lucide-react)
    ├── vite.config.js
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── spec.md                   ← Spec UI chi tiết
    └── src/
        ├── main.jsx              ← React bootstrap
        ├── App.jsx               ← Root component (state: panels, results, topK)
        ├── index.css             ← Global CSS + Tailwind directives
        ├── data/
        │   └── mockResults.js    ← Dữ liệu mock (khi backend unavailable)
        └── components/
            ├── Header.jsx        ← Logo K2PN + Settings(TopK) + Reset + User badge
            ├── Sidebar.jsx       ← Danh sách SearchPanels + +/- + Search button
            ├── SearchPanel.jsx   ← 1 panel: icons Text/ASR/OCR/Obj + textarea + toggle
            ├── ResultGrid.jsx    ← Lưới ảnh responsive 2–7 cột
            └── ResultDetailModal.jsx ← Modal xem chi tiết ảnh lớn + metadata
```

---

## BẰNG 1: BACKEND (FastAPI) Chi tiết

### 1. Khởi động nhanh

```bash
# 1. Terminal 1 — Backend FastAPI (:8888)
conda activate aic2026_backend
cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd

# (Option A — tmux, khuyến nghị cho production)
tmux new -s backend
uvicorn main:app --host 0.0.0.0 --port 8888

# (Option B — dev quick)
python main.py
```

> **Lưu ý GPU**: Line đầu file `main.py` hardcode `CUDA_VISIBLE_DEVICES="6"` (trước khi import torch). Nếu muốn dùng GPU khác: `CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port 8888` (override env trước khi start).

Đợi log `"=== Models loaded ✓ ==="` — lúc đó backend sẵn sàng. Kiểm tra:
```bash
curl http://localhost:8888/health
# → {"status":"ok","models_loaded":true,"qdrant_connected":true}
```

### 2. Models đang sử dụng

| Model | HuggingFace ID | Status | Dim | Library | GPU dtype |
|---|---|---|---|---|---|
| **SigLIP2** | `google/siglip2-base-patch16-224` | ✅ **BẬT mặc định** | 768 | `transformers` AutoModel | float16 |
| **PE-Core-L** | `hf-hub:timm/PE-Core-L-14-336` | ❌ **TẮT mặc định** | 1024 | `open_clip` | float16 |

**Cách bật/tắt PE-Core-L**: Sửa flag `USE_PECORE = True/False` ở [main.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/main.py#L47-L47).

### 3. SigLIP2 Encoder — chi tiết kỹ thuật quan trọng

Định nghĩa ở [models/siglip_encoder.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/models/siglip_encoder.py).

**Vấn đề HuggingFace SigLipModel**: `SiglipModel.forward()` yêu cầu CẢ text + image inputs cùng lúc (vision model sẽ crash nếu `pixel_values=None`). Hàm `get_text_features()` chỉ chạy text branch, **KHÔNG qua projection head** → gãy alignment không gian contrastive.

**Giải pháp hiện tại**:
1. Text encode thật: `input_ids` thực
2. Tạo `pixel_values` = DUMMY zero-tensor shape `(B, 3, 224, 224)` cùng dtype + device
3. Gọi `siglip_model(input_ids=..., pixel_values=..., return_loss=False)`
4. Lấy `outputs["text_embeds"][0]` — đây là embedding ĐÃ qua projection head và ĐÃ L2-normalized
5. `.cpu().float().numpy()` → shape `(768,)` float32 cho Qdrant

Image encode cũng tương tự (dummy text token pad_id=1).

### 4. Qdrant Search Config

Định nghĩa ở [search/qdrant_search.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/search/qdrant_search.py).

| Config | Giá trị |
|---|---|
| Host | `127.0.0.1` (localhost) |
| Port | `6333` |
| Client timeout (init) | `120s` |
| Per-query timeout | `120s` |
| Client pattern | **Singleton lazy** — `get_qdrant_client()` tạo 1 lần đầu tiên gọi |
| Query API | `client.query_points()` (API mới, qdrant-client v1.8+) |
| Vector field | `unnamed` (default, không cần `using=`) |
| Distance metric | **Cosine** |

**Collections hiện có**:

| Collection name | Dim | Model nguồn | Trạng thái |
|---|---|---|---|
| `image_siglip` | 768 | google/siglip2-base-patch16-224 | ✅ active |
| `image_pecore` | 1024 | timm/PE-Core-L-14-336 | ⚠️ active nếu `USE_PECORE=True` |

**Payload structure** (cả 2 collection):
```json
{
  "video_id":    "L24_V005",
  "keyframe_id": "L24_V005_kf_0187",
  "frame_idx":   187,
  "image_path":  "L24_V005/L24_V005_kf_0187.jpg"
}
```

**Filter video (optional)**: Nếu `video_list` không rỗng trong request →:
```python
Filter(must=[FieldCondition(key="video_id", match=MatchAny(any=video_list))])
```

### 5. RRF Fusion Config

Định nghĩa ở [search/rrf.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/search/rrf.py).

| Param | Giá trị |
|---|---|
| Default `rrf_k` (backend) | `60` (được override nếu frontend gửi) |
| `fetch_depth` (pre-fuse) | `max(100, top_k * 5)` |
| Score normalize | `YES` — chia cho `(num_active_models) * (1/(k+1))`, round 6 dp → range (0,1] |
| Key dedupe | `keyframe_id` (unique toàn DB) |

### 6. FastAPI Endpoints

| Method | Route | Vai trò |
|---|---|---|
| `GET` | `/` | Serve `Test_UI/index.html` (frontend standalone cũ) |
| `GET` | `/health` | Trạng thái server + models loaded + Qdrant kết nối |
| `POST` | `/api/search` | Nhận truy vấn → encode → query Qdrant → RRF → trả kết quả |
| `GET` | `/images/{video_id}/{filename}` | Serve static keyframe JPG |

### 7. Request Schema (`POST /api/search`)

Định nghĩa ở [schemas.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/schemas.py).

```json
{
  "query":      "cô gái mặc áo đỏ đi xe máy",   // bắt buộc, không rỗng (auto strip)
  "top_k":       20,                              // bắt buộc, > 0
  "rrf_k":       60,                              // tùy chọn, default 60, >= 1
  "video_list":  ["L01_V001", "L02_V005"],        // tùy chọn, []/None = toàn DB
  "use_ocr":     false,                           // placeholder — chưa implement
  "use_asr":     false                            // placeholder — chưa implement
}
```

### 8. Response Schema (`POST /api/search`)

```json
{
  "query": "cô gái mặc áo đỏ đi xe máy",
  "top_k": 20,
  "rrf_k": 60,
  "results": [
    {
      "rank":       1,
      "score":      0.923754,
      "video_id":   "L24_V005",
      "keyframe_id":"L24_V005_kf_0187",
      "frame_idx":  187,
      "image_url":  "/images/L24_V005/L24_V005_kf_0187.jpg"   // root-relative
    }
  ]
}
```

### 9. Concurrency & Performance

| Thành phần | Cấu hình |
|---|---|
| Model lifecycle | Load 1 lần ở **lifespan startup** (không load per-request) |
| Thread pool for blocking encode/query | `ThreadPoolExecutor(max_workers=2)` |
| Encode parallel (text x 2 model) | `asyncio.gather(pecore_future, siglip_future)` (nếu USE_PECORE) |
| Query parallel (Qdrant x 2 collection) | `asyncio.gather(pecore_query, siglip_query)` |
| Logging | `logging.INFO` format `HH:MM:SS \| LEVEL \| name \| msg` |
| CORS | `allow_origins=["*"]` + `allow_methods=["*"]` + `allow_headers=["*"]` |

### 10. Static file paths

| Config | Path |
|---|---|
| Keyframes root vật lý | `/AIClub_NAS/core_baotg/nhan/dataset/keyframes` |
| FastAPI mount route | `/images` |
| `image_url` trả về | `/images/{image_path_from_payload}` |

---

## BẰNG 2: FRONTEND (React + Vite) Chi tiết

### 1. Khởi động nhanh

```bash
# Cách A (đơn giản nhất — không cần cài gì thêm):
#   Mở trình duyệt → http://<SERVER_IP>:8888/
#   Backend serve trực tiếp index.html CDN standalone (chỉ cần backend đang chạy)

# Cách B (Vite dev mode — hot reload, dùng khi phát triển UI):
cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/Test_UI
npm install                # 1 lần đầu
npm run dev                # Vite dev server (default :5173)
# Vite sẽ tự động gọi backend http://localhost:8888

# Cách C (Build production):
npm run build              # → dist/ folder
# Sau đó: copy dist/index.html thay cho Test_UI/index.html cũ
# Hoặc dùng nginx/caddy serve dist/, API proxy pass sang :8888
```

### 2. Công nghệ & Stack

| Thư viện | Version | Vai trò |
|---|---|---|
| React | 18.3.1 | UI framework (function components + hooks) |
| React-DOM | 18.3.1 | DOM renderer |
| Vite | 5.4.1 | Dev server + build tool |
| @vitejs/plugin-react | 4.3.1 | JSX/Babel transform |
| Tailwind CSS | 3.4.4 | Utility-first CSS |
| PostCSS | 8.4.40 | CSS processor |
| Autoprefixer | 10.4.19 | Cross-browser CSS compat |
| lucide-react | 0.484.0 | Icon library (Text, Mic, FileText, Image, Play, RotateCcw, Settings...) |

### 3. API Base URL

File [App.jsx](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/Test_UI/src/App.jsx#L8-L8):
```jsx
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8888';
```
→ **Nếu cần đổi backend host/port**: set env var `VITE_API_BASE_URL` trước khi build, hoặc sửa default trực tiếp.

### 4. Trạng thái ứng dụng (App-level state)

Ở component [App.jsx](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/Test_UI/src/App.jsx#L17-L25):

```jsx
const [panels, setPanels]           // SearchPanel[] (ít nhất 1)
const [results, setResults]         // ResultItem[] (mapped từ backend JSON)
const [selectedResult, setSelectedResult] // ResultItem mở modal
const [isSearching, setIsSearching] // bool — spinner
const [topK, setTopK]               // int (default 10)
const [apiError, setApiError]       // null | string (banners dưới phải)
const [searchError, setSearchError] // null | string (ResultGrid empty state)
const [backendStatus, setBackendStatus] // {reachable, models_loaded} — health badge
```

### 5. Component Tree

```
App (src/App.jsx)
├── Header             (src/components/Header.jsx)
│   ├── Logo "K2PN"
│   ├── Settings Popup
│   │   └── TopK number input (prop-driven: nhận topK + onChangeTopK)
│   ├── Refresh (Reset) button
│   └── User "U" badge
├── Backend Health Badge (inline JSX, App.jsx)
├── Sidebar            (src/components/Sidebar.jsx)
│   ├── Header: Panel count + Enabled count
│   ├── <SearchPanel> (map() theo panels[])
│   │     ├── SearchType icons row:  Text / ASR / OCR / Object
│   │     ├── Query textarea (rows=3)
│   │     └── Toggle Enable switch
│   ├── Panel controls: [+ Thêm] [- Xóa cuối]
│   └── [Thực hiện tìm kiếm] Submit button
├── ResultGrid         (src/components/ResultGrid.jsx)
│   ├── Loading spinner (nếu isSearching)
│   ├── Error empty state
│   ├── Empty state (chưa search)
│   └── Grid (cols: 2 sm:3 md:4 lg:6 xl:7)
│         └── <Card> (rank badge, match type badge, hover play overlay, caption + videoId)
├── API Error Banner (fixed bottom-right)
└── ResultDetailModal  (src/components/ResultDetailModal.jsx) — khi click card
      ├── Big image (object-contain)
      ├── Close button
      ├── Video title, caption
      └── Metadata rows: FrameId, Rank, MatchedType
```

### 6. Luồng handleSearch (App.jsx)

1. `setIsSearching(true)`, reset errors
2. Lọc `activePanels` (`enabled` + `query` không rỗng), lấy mảng `queries`
3. **Nếu không có query nào** → hiển thị `MOCK_RESULTS` (dữ liệu nội bộ mockResults.js)
4. Ngược lại, build payload:
   ```json
   {
     "query": queries.join(" | "),
     "top_k": topK,
     "rrf_k": 60,
     "video_list": [],
     "use_ocr": activePanels.some(p=>p.type==='ocr'),
     "use_asr": activePanels.some(p=>p.type==='asr')
   }
   ```
5. `fetch(API_BASE_URL + "/api/search")` method `POST`
6. Map backend results → `ResultItem` (resolve `image_url` absolute URL via `window.location.origin`)
7. Catch lỗi → set `apiError` / `searchError`
8. `finally { setIsSearching(false) }`

### 7. Health Check (mỗi lần mount App)

- Gọi `GET {API_BASE_URL}/health` 1 lần duy nhất ở `useEffect([])`
- Update badge:
  - 🔴 `Unavailable (using mock)` — không connect được backend
  - 🟡 `Live (loading models)` — connect OK nhưng models chưa xong
  - 🟢 `Live (models ready)` — backend sẵn sàng

---

## BẰNG 3: Quy trình Offline Indexing (.npy → Qdrant)

Khi có batch keyframes mới hoặc model/version thay đổi:

```bash
cd /AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd/tests

# Bước A: Encode toàn bộ keyframes → .npy files
#          (Pipeline offline riêng, thường ở src/preprocessing)

# Bước B: Fast import .npy → Qdrant (đã có sẵn)
python fast_import_npy_to_qdrant.py

# Bước C: Kiểm tra collection sau khi import
python check_qdrant_collection.py

# Bước D: Sanity check (encode text query, verify top score hợp lý)
python run_sanity_check.py
```

---

## Ghi chú deployment

- **Port cần mở / tunnel**: `8888` (FastAPI + images + index.html serve)
- **Đối với SSH tunnel**: `ssh -L 8888:localhost:8888 user@server-ip` → mở trình duyệt `http://localhost:8888/`
- **Qdrant không cần expose ngoài** — backend gọi localhost:6333 nội bộ
- **Models cache HF**: `~/.cache/huggingface/hub/` — đảm bảo đủ ổ cứng (1-2 GB mỗi model)
