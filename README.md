# AIC 2026 — Interactive Video Search System

Hệ thống Tìm kiếm Video Thông minh cho Cuộc thi **AI Challenge (AIC) 2026**, hỗ trợ truy vấn đa thức (Text-to-Image, OCR, ASR, Temporal Search, RRF Fusion) trên tập dữ liệu Video quy mô lớn.

---

## 📌 1. Tổng Quan Dự Án & Kiến Trúc

Hệ thống được thiết kế theo quy trình mô-đun hóa 7 bước (7-Stage Pipeline) để xử lý dữ liệu thô từ video thành các chỉ mục tìm kiếm văn bản và vector:

```
[Raw Videos] ➔ 1. TransNetV2 (Shot Cut) ➔ 2. OpenCLIP (Keyframe Filter)
            ➔ 3. PaddleOCR (OCR Text)   ➔ 4. Faster-Whisper (ASR Speech)
            ➔ 5. SigLIP2 (Vector Embed) ➔ 6. Elasticsearch ➔ 7. Qdrant
                                                │                │
                                                └───────┬────────┘
                                                        ▼
                                           [FastAPI Backend Server]
```

---

## 🛠️ 2. Mô Hình & Công Cụ Sử Dụng (Models & Tech Stack)

| Thành phần | Mô hình / Công nghệ | Mô tả & Cấu hình |
| :--- | :--- | :--- |
| **Shot Cut** | **TransNetV2** | Phát hiện ranh giới chuyển cảnh video chính xác |
| **Keyframe Filtering** | **OpenCLIP `ViT-L-14-quickgelu`** | Pretrained `dfn2b` (Vortex paper), L2-relative diff `threshold=0.4`, `step=8` |
| **OCR (Trích xuất chữ)** | **PaddleOCR** | Trích xuất chữ tiếng Việt và bounding polygon từ keyframes |
| **ASR (Trích xuất tiếng)** | **Faster-Whisper** | Phân giải giọng nói tiếng Việt & tự động ánh xạ cửa sổ 6 giây xung quanh keyframe |
| **Image Embedding** | **SigLIP2 (`so400m-patch14-384`)** | Trích xuất vector đặc trưng hình ảnh 1152 chiều |
| **Text Search Engine** | **Elasticsearch 9.x** | Chỉ mục tìm kiếm văn bản tổng hợp (OCR, ASR, Captions, Objects) |
| **Vector Search Engine** | **Qdrant** | Collection `image_siglip`, vector 1152-dim FLOAT32 Full Recall, HNSW (`m=32`, `ef_construct=512`) |
| **Web API Backend** | **FastAPI + Uvicorn** | Khởi chạy server RESTful API tìm kiếm phản hồi thời gian thực |

---

## 🚀 3. Hướng Dẫn Cài Đặt & Khởi Chạy

### 3.1. Thiết lập Môi Trường (Conda Environment)

Kích hoạt môi trường Conda đã được chuẩn bị sẵn trên server:
```bash
conda activate aic2026_backend
```

Hoặc tạo môi trường mới từ file [`environment.yml`](environment.yml):
```bash
conda env create -f environment.yml
conda activate aic2026_backend
```

---

### 3.2. Cấu Hình Dataset (HCMAI 2026 Batch 1)

Dự án sử dụng tập dữ liệu **HCMAI 2026 Batch 1**.

Chỉnh sửa đường dẫn dữ liệu đầu vào và đầu ra trong file cấu hình [`configs/runtime.yaml`](configs/runtime.yaml):

```yaml
default:
  mode: full
  format: csv
  input_dir: /mlcv2025/Datasets/HCMAI25/batch1/video    # Đường dẫn thư mục chứa video Batch 1
  output_dir: /AIClub_NAS/core_baotg/phong/AIC_2026/outputs  # Thư mục chứa kết quả
```

---

### 3.3. Khởi Chạy Quy Trình Preprocessing (7-Stage Pipeline)

Chạy toàn bộ quy trình tiền xử lý và tạo chỉ mục tìm kiếm từ Stage 1 đến Stage 7:

```bash
# Chạy toàn bộ 7 bước liên hoàn:
python src/preprocessing/run_pipeline.py --mode full --stages all
```

Hoặc chạy các bước riêng lẻ nếu cần:
```bash
# Ví dụ chỉ chạy Stage 3 OCR và Stage 5 Embedding:
python src/preprocessing/run_pipeline.py --mode full --stages 3,5

# Ví dụ chỉ chạy nạp chỉ mục Elasticsearch (Stage 6) và Qdrant (Stage 7):
python src/preprocessing/run_pipeline.py --mode full --stages 6,7
```

---

### 3.4. Khởi Động BackEnd Server (FastAPI)

Sau khi tạo chỉ mục xong, chuyển vào thư mục Backend và khởi động server Uvicorn:

```bash
cd UI/BackEnd
uvicorn main:app --host 0.0.0.0 --port 8888
```

Server Backend sẽ lắng nghe tại: `http://0.0.0.0:8888`.
Thao tác kiểm tra API Docs Swagger tại: `http://localhost:8888/docs`.
### 👥 4. Thành Viên Dự Án (Contributors)
Nguyễn Hải Phong: github.com/sherwyndd
Lê Đức Nhân: github.com/nhanlenhanle
Dương Anh Kiệt: 
Võ Hoàng Kim: 
