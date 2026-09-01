# 📊 BÁO CÁO TỔNG QUAN HỆ THỐNG TÌM KIẾM VIDEO AI CHALLENGE 2026 (K2PN)
> **Phân tích Kiến trúc, Kỹ thuật và Các Mô hình AI (Models)**  
> *Dựa trên phân tích mã nguồn hệ thống Giao diện (`UI/Test_UI/index.html`), Backend (`UI/BackEnd`), và Offline Processing Pipeline.*

---

## 📌 MỤC LỤC
1. [Tổng quan Kiến trúc Hệ thống (System Architecture)](#1-tổng-quan-kiến-trúc-hệ-thống-system-architecture)
2. [Chi tiết các Mô hình AI & Deep Learning (Models)](#2-chi-tiết-các-mô-hình-ai--deep-learning-models)
3. [Các Kỹ thuật và Thuật toán Cốt lõi (Core Techniques & Algorithms)](#3-các-kỹ-thuật-và-thuật-toán-cốt-lõi-core-techniques--algorithms)
   - [3.1. Truy vấn Vector & Text đa nguồn](#31-truy-vấn-vector--text-đa-nguồn)
   - [3.2. Dung hợp kết quả đa nguồn (Adaptive RRF)](#32-dung-hợp-kết-quả-đa-nguồn-adaptive-rrf)
   - [3.3. Thuật toán tìm kiếm theo chuỗi thời gian (Temporal & TraKE)](#33-thuật-toán-tìm-kiếm-theo-chuỗi-thời-gian-temporal--trake)
   - [3.4. Mở rộng khung hình lân cận (Surrounding Frames / Radius Expansion)](#34-mở-rộng-khung-hình-lân-cận-surrounding-frames--radius-expansion)
   - [3.5. Kỹ thuật tối ưu hiệu năng & Low-Latency I/O](#35-kỹ-thuật-tối-ưu-hiệu-năng--low-latency-io)
4. [Các Phân hệ Tính năng Chuyên sâu trên Giao diện (`index.html`)](#4-các-phân-hệ-tính-năng-chuyên-sâu-trên-giao-diện-indexhtml)
   - [4.1. Workspace Tìm kiếm Đa phương thức (Multimodal Panels)](#41-workspace-tìm-kiếm-đa-phương-thức-multimodal-panels)
   - [4.2. Phân hệ TraKE (Tracking Keyframe & Event Search)](#42-phân-hệ-trake-tracking-keyframe--event-search)
   - [4.3. Phân hệ Stage-based Temporal Search](#43-phân-hệ-stage-based-temporal-search)
   - [4.4. Phân hệ Check Video & Video Streaming Player](#44-phân-hệ-check-video--video-streaming-player)
   - [4.5. Phân hệ Quản lý Kết quả, Nộp bài DRES & Cộng tác](#45-phân-hệ-quản-lý-kết-quả-nộp-bài-dres--cộng-tác)
5. [Tổng kết & Đánh giá](#5-tổng-kết--đánh-giá)

---

## 1. Tổng quan Kiến trúc Hệ thống (System Architecture)

Hệ thống **K2PN** được thiết kế theo mô hình **End-to-End Multimodal Video Retrieval System**, chia thành 3 phân tầng xử lý tách biệt:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│ 1. OFFLINE PIPELINE (Data Preprocessing & Indexing)                           │
│  • Shot Cut (TransNetV2) ➔ Keyframe Extraction (every-8th + CLIP L2 filter)    │
│  • ASR Extraction (Whisper large-v3) ➔ OCR (PaddleOCR/PARSeq) ➔ OBJ (YOLOv8)  │
│  • Offline Embedding ➔ Vector DB (Qdrant) & Text Search Engine (Elasticsearch) │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────────┐
│ 2. BACKEND LAYER (FastAPI - Port 8888)                                         │
│  • Multimodal Encoders: SigLIP2 (768D) + PE-Core-L (1024D OpenCLIP)            │
│  • Vector Search: Qdrant HNSW FP32 Cosine Search                              │
│  • Text Search: Elasticsearch BM25 (OCR, ASR)                                 │
│  • Hybrid Fusion: Adaptive RRF (Reciprocal Rank Fusion) + BGE Reranker         │
│  • Temporal & TraKE Engine: Beam Search DP + Action-Distance NMS               │
│  • High-performance Caching: Preload ~350k Keyframes & 708 ASR JSONL vào RAM   │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼─────────────────────────────────────────┐
│ 3. FRONTEND UI (Test_UI/index.html - React 18 Standalone + TailwindCSS)       │
│  • Multi-modal Query Panels (Semantic Text, OCR, ASR, Image Similarity)       │
│  • TraKE & Multi-Stage Temporal Search Timeline Workspace                     │
│  • Video Deep Inspection (Sub-second PTS mapping, Scrubber, Synchronized ASR)  │
│  • Shared Ranking Workspace (KIS, QA, TraKE Top-100 Auto-Fill & DRES Submit)   │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Chi tiết các Mô hình AI & Deep Learning (Models)

| Tác vụ | Tên Model & HuggingFace ID | Thông số & Chi tiết Kỹ thuật | Vai trò trong Hệ thống |
| :--- | :--- | :--- | :--- |
| **Vision-Language Dense Embedding** | `google/siglip2-base-patch16-224` | • Kích thước vector: **768-dim**<br>• Framework: `transformers.AutoModel`<br>• Precision: `float16` trên GPU (`cuda:0`)<br>• Metric: Cosine Similarity | **Model chủ lực (Primary Dense Retriever)**: Mã hóa câu truy vấn văn bản và ảnh keyframe vào cùng không gian đa phương thức. Áp dụng kỹ thuật chèn dummy tensor để đi qua projection head và chuẩn hóa $L_2$. |
| **Ensemble Dense Visual Encoder** | `hf-hub:timm/PE-Core-L-14-336` | • Kích thước vector: **1024-dim**<br>• Framework: `open_clip`<br>• Precision: `float16` | **Model phụ trợ**: Hỗ trợ ensemble làm giàu góc nhìn visual, giảm thiểu bias của mô hình đơn lẻ. |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | • Framework: `FlagEmbedding / transformers`<br>• Precision: `float16` | **Reranking Top-N**: Đánh giá tương quan ngữ nghĩa chuyên sâu giữa query và ứng viên sau bước fusion để tối ưu hóa Top-1 / Top-5 Precision. |
| **Speech-to-Text (ASR)** | `OpenAI Whisper (large-v3)` | • Architecture: Encoder-Decoder Transformer<br>• Output: Timestamped JSONL | **Bóc tách lời thoại**: Trích xuất toàn bộ hội thoại/âm thanh trong video kèm timestamp chính xác, chia nhỏ thành cửa sổ trượt 6 giây (`asr_6s`). |
| **Optical Character Recognition (OCR)** | `PaddleOCR` / `PARSeq` | • Detection: DBNet / TextDetector<br>• Recognition: SVTR / PARSeq | **Bóc tách chữ trên màn hình**: Nhận diện biển hiệu, phụ đề, nhãn đồ vật trên từng khung hình, lập chỉ mục vào Elasticsearch. |
| **Object Detection** | `YOLOv8` (`yolov8s.pt`) / `Co-DETR` | • Framework: `ultralytics`<br>• Backbone: CSPDarknet | **Phát hiện vật thể**: Trích xuất nhãn và bounding box của các đối tượng trong keyframe để hỗ trợ lọc và truy vấn theo đối tượng. |
| **Shot Cut & Boundary Detection** | `TransNetV2` | • Input: Video 3D Conv frames<br>• Output: Cut / Transition probabilities | **Phân đoạn video**: Nhận diện ranh giới các cảnh quay (hard cut & gradual transitions) làm cơ sở trích xuất keyframe đại diện. |

---

## 3. Các Kỹ thuật và Thuật toán Cốt lõi (Core Techniques & Algorithms)

### 3.1. Truy vấn Vector & Text đa nguồn
* **Qdrant HNSW Vector Search**:
  * Chỉ mục đồ thị phân tầng: $M=32, ef\_construct=512, ef\_search=256$.
  * Toàn vẹn dữ liệu FP32 không nén (Full-precision) để đảm bảo tối đa độ nhạy Recall.
  * **Dynamic Oversampling**: `qdrant_fetch_depth = max(top_k * 10, 100)` giúp giữ lại các True Positives trước khi bước vào dung hợp đa nguồn.
* **Elasticsearch BM25**:
  * Đánh chỉ số văn bản cho trường ASR và OCR với `minimum_should_match` linh hoạt (50% - 60%).
  * Hỗ trợ tìm kiếm từ khóa cục bộ, khớp từng phần và xử lý tiếng Việt.

### 3.2. Dung hợp kết quả đa nguồn (Adaptive RRF - Reciprocal Rank Fusion)
Hệ thống sử dụng **Adaptive Reciprocal Rank Fusion** thay cho Min-Max Normalization để triệt tiêu ảnh hưởng của điểm số bất định (unbounded outliers) từ BM25:

$$\text{Score}_{\text{RRF}}(d) = \sum_{s \in \text{Sources}} \frac{w_s}{k_s + \text{Rank}_s(d)}$$

* **Tham số làm mịn thích ứng ($k_s$) theo từng nguồn**:
  * **Semantic Search (SigLIP2)**: $k \approx 40 - 50$ (tăng trọng số cho các vị trí top đầu có độ tin cậy cao).
  * **ASR & OCR (Elasticsearch)**: $k \approx 80 - 100$ (làm phẳng phân phối điểm, tăng độ bao phủ và đa dạng).
  * **Object Search**: $k \approx 60$.
* **Text-Image Hybrid Fusion**: Kết hợp điểm ảnh tương đồng và mô tả văn bản bổ trợ theo công thức:
  $$\text{Score}_{\text{Fused}} = 0.75 \times \text{Score}_{\text{Text}} + 0.25 \times \text{Score}_{\text{Image}}$$

### 3.3. Thuật toán tìm kiếm theo chuỗi thời gian (Temporal & TraKE)
* **Beam Search Dynamic Programming (DP)**:
  * Căn chỉnh chuỗi sự kiện theo thứ tự thời gian nghiêm ngặt:
    $$\text{Timestamp}(E_0) < \text{Timestamp}(E_1) < \dots < \text{Timestamp}(E_{N-1})$$
* **Ràng buộc khoảng cách (Gap Constraints & Penalty)**:
  * Áp dụng `max_gap_seconds` giữa các stage liên tiếp.
  * Áp dụng hệ số phạt khoảng cách thời gian (Temporal Distance Penalty) nhằm ưu tiên các chuỗi hành động diễn ra liền mạch.
* **Global Context Bonus**: Thưởng điểm cho toàn bộ chuỗi nếu bối cảnh chung của video khớp với truy vấn ngữ cảnh toàn cục (`global_context`).
* **Action-Distance NMS (Non-Maximum Suppression)**: Loại bỏ các chuỗi keyframe bị trùng lặp trong cùng phân cảnh để đảm bảo danh sách kết quả chứa các video đa dạng nhất.

### 3.4. Mở rộng khung hình lân cận (Surrounding Frames / Radius Expansion)
* Cho phép mở rộng nhanh các frame lân cận trong bán kính $\pm N$ ($N \in [1, 10]$) xung quanh keyframe tìm được.
* Tự động tái tính toán `pts_time` (giây) và `timestamp` (mili-giây) theo FPS chính xác của video, đảm bảo không bỏ sót khoảnh khắc chuẩn xác của đề bài.

### 3.5. Kỹ thuật tối ưu hiệu năng & Low-Latency I/O
* **Zero-Disk I/O Pipeline**: Preload toàn bộ metadata của hơn **350.000 keyframes** và **708 file JSONL ASR** vào RAM khi server khởi động, chuyển đổi toàn bộ tác vụ tra cứu timestamp/FPS thành tra cứu Hash Map O(1).
* **Bất đồng bộ hóa (Async Concurrency)**: Tách riêng luồng xử lý `query_executor` (tính toán GPU, vector search) và `io_executor` (đọc file/metadata) thông qua `ThreadPoolExecutor` và `asyncio.gather`.
* **Aggressive HTTP Caching**: Thiết lập header `Cache-Control: public, max-age=31536000, immutable` cho kho ảnh keyframe tĩnh.

---

## 4. Các Phân hệ Tính năng Chuyên sâu trên Giao diện (`index.html`)

File [index.html](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/Test_UI/index.html) là một ứng dụng **React 18 Single-Page Application** hoàn chỉnh (hơn 9.700 dòng lệnh), tích hợp trực tiếp:

### 4.1. Workspace Tìm kiếm Đa phương thức (Multimodal Panels)
* Thêm/bớt không giới hạn các thẻ tìm kiếm: **Text (Semantic)**, **OCR**, **ASR**, **Image Similarity (Keyframe ID)**.
* Thiết lập trọng số độc lập cho từng thẻ tìm kiếm, kích hoạt/vô hiệu hóa từng thẻ linh hoạt.
* Hiển thị lưới ảnh kết quả Responsive với Skeleton Loading, Lazy Loading và bộ đếm thời gian truy vấn theo thời gian thực.

### 4.2. Phân hệ TraKE (Tracking Keyframe & Event Search)
* Mô hình hóa chuỗi sự kiện ($E_1, E_2, E_3 \dots$) với từng mô tả text, OCR, ASR và thanh gạt trọng số riêng.
* Hiển thị chuỗi timeline trực quan của từng video kèm khoảng thời gian bao phủ (`time_span`, `duration_sec`).
* **TrakeVideoInspectorModal**: Modal chuyên sâu cho phép xem chuỗi sự kiện, duyệt qua các frame lân cận và thay thế frame trong chuỗi.

### 4.3. Phân hệ Stage-based Temporal Search
* Thiết kế truy vấn theo từng Stage liên tiếp, mỗi Stage hỗ trợ nhiều Sub-queries đa phương thức.
* Cấu hình trọng số cấp Stage (`stage_weight`) và giới hạn khoảng cách thời gian giữa các Stage (`max_gap_seconds`).
* Xử lý triệt tiêu Race Condition khi gửi truy vấn liên tục bằng `AbortController`.

### 4.4. Phân hệ Check Video & Video Streaming Player
* Phát video gốc trực tiếp từ backend qua cơ chế stream MP4 (`/api/video/{video_id}/stream`).
* Thanh trượt Scrubber với độ phân giải khung hình cao, tự động chuyển đổi qua lại giữa Frame Index $\leftrightarrow$ PTS Timestamp.
* Đồng bộ hiển thị dòng phụ đề ASR theo thời gian thực và dải thumbnail toàn bộ keyframe của video.

### 4.5. Phân hệ Quản lý Kết quả, Nộp bài DRES & Cộng tác
* **3 Bảng xếp hạng chuyên dụng**:
  1. **KIS Ranking**: Quản lý danh sách kết quả cho bài toán Known-Item Search.
  2. **QA Ranking**: Quản lý kết quả trả lời câu hỏi, cho phép nhập đáp án trực tiếp cho từng frame.
  3. **TraKE Ranking**: Quản lý danh sách các chuỗi video/sự kiện.
* **Auto-Fill Top 100**: Tự động điền đầy danh sách 100 kết quả từ lượt tìm kiếm gần nhất.
* **Jump Rank**: Nhảy nhanh kết quả đến thứ hạng chỉ định (#1 - #100).
* **DRES Submission & CSV Export**: Tích hợp nút nộp bài trực tiếp lên máy chủ chấm điểm DRES và xuất file CSV chuẩn định dạng nộp bài của cuộc thi.
* **WebSocket Real-time Collaboration**: Thiết kế sẵn cơ chế đồng bộ clipboard theo thời gian thực giữa các thành viên trong nhóm.

---

## 5. Tổng kết & Đánh giá

Hệ thống **K2PN (AI Challenge 2026)** là một giải pháp truy vấn video toàn diện, kết hợp chặt chẽ giữa:
1. **Sức mạnh mô hình AI**: Sử dụng các mô hình thị giác - ngôn ngữ tiên tiến nhất (**SigLIP2, Whisper large-v3, BGE-Reranker-v2-m3**).
2. **Kỹ thuật tìm kiếm kết hợp**: Kết hợp Vector Search (Qdrant) và Text Search (Elasticsearch) thông qua **Adaptive RRF**.
3. **Căn chỉnh thời gian thông minh**: Áp dụng **Dynamic Programming Sequence Alignment** cho các bài toán truy vết hành động phức tạp.
4. **Trải nghiệm người dùng tối ưu**: Giao diện [index.html](file:///AIClub_NAS/core_baotg/phong/AIC_2026/UI/Test_UI/index.html) trực quan, độ trễ thấp, hỗ trợ tối đa tốc độ thao tác cho thí sinh trong các vòng thi đấu thời gian thực.

---

## 6. Đánh giá Hạn chế & Giải pháp Tối ưu Quá trình trích xuất Keyframe

Quá trình trích xuất Keyframe (Stage 2: `02_keyframe_ext.py` & `clip_embedder.py`) hiện tại sử dụng cơ chế **Candidate Sampling (mỗi 8 frames)** kết hợp **CLIP L2 Filtering (loại bỏ khung hình có độ lệch tương đối < 0.4)**. Mặc dù thông minh hơn các phương pháp cũ, cơ chế này vẫn có 3 hạn chế cốt lõi và các giải pháp đề xuất tương ứng:

### 6.1. Sự bất đồng bộ về Không gian Vector (Model Mismatch)
* **Hạn chế**: Quá trình lọc Keyframe offline dùng mô hình `openai/clip-vit-large-patch14`, nhưng cỗ máy Semantic Search ở Backend lại dùng `google/siglip2-base-patch16-224`. Sự khác biệt về kiến trúc khiến hai mô hình đánh giá "sự khác biệt hình ảnh" không giống nhau. CLIP có thể loại bỏ 1 frame mà SigLIP2 lại coi là quan trọng, dẫn đến việc mất mát dữ liệu tìm kiếm (mất True Positives).
* **Giải pháp khắc phục**: 
  * Thay thế mô hình `clip-vit-large-patch14` trong file `src/pipeline/clip_embedder.py` bằng chính mô hình **`google/siglip2-base-patch16-224`**. Việc đồng bộ 100% mô hình giữa khâu Preprocessing và khâu Search sẽ đảm bảo các keyframe được giữ lại là những keyframe mà SigLIP2 đánh giá là khác biệt nhất.

### 6.2. Rủi ro mất các hành động diễn ra nhanh (Fixed Sample Step)
* **Hạn chế**: Việc cố định lấy mẫu `sample_step = 8` (tương đương ~0.33 giây/frame với video 24fps) tạo ra các "khoảng mù". Nếu một hành động quan trọng (ví dụ: một vụ nổ nhỏ, một tờ giấy bay ngang) chỉ kéo dài trong 3-4 frame và rơi vào giữa khoảng trống của `sample_step`, hệ thống sẽ hoàn toàn bỏ lỡ nó.
* **Giải pháp khắc phục**:
  * **Dynamic Sampling dựa trên Optical Flow hoặc Pixel Diff**: Trước khi đưa qua mô hình AI, tính toán nhanh cường độ chuyển động (Motion Intensity) giữa các frame liên tiếp bằng các thuật toán siêu nhẹ (như SSIM, Frame Difference). Nếu phát hiện đoạn video có chuyển động mạnh, tự động giảm `sample_step` xuống 2 hoặc 4. Nếu video tĩnh, tăng `sample_step` lên 12 hoặc 24.
  * Hoặc đơn giản là giảm `sample_step = 4` kết hợp với tăng nhẹ `rel_diff_threshold = 0.45` để bắt được các cảnh nhanh nhưng không làm tăng đột biến số lượng keyframe.

### 6.3. Hao phí tài nguyên tính toán không cần thiết (Overhead)
* **Hạn chế**: Việc chạy một mô hình cực nặng như `clip-vit-large` trên *hàng ngàn candidate* của mỗi video, chỉ để vứt đi phần lớn trong số chúng (do giống nhau), là một sự lãng phí tài nguyên GPU và thời gian trích xuất khổng lồ.
* **Giải pháp khắc phục**:
  * **Áp dụng Bộ lọc Cascade (Cascade Filtering)**:
    1. **Pass 1 (Siêu nhẹ - CPU)**: Dùng thuật toán Hash hình ảnh (Perceptual Hash - pHash) hoặc tính toán SSIM để gạt bỏ ngay lập tức các frame giống hệt nhau về mặt pixel (ví dụ: video tĩnh, slide bài giảng).
    2. **Pass 2 (Nặng - GPU)**: Chỉ đưa các frame đã vượt qua Pass 1 vào mô hình SigLIP2 (hoặc CLIP) để trích xuất vector và lọc tiếp bằng L2 Distance như hiện tại. Việc gạt bỏ sớm ở Pass 1 sẽ giảm từ 40-60% lượng việc mà GPU phải xử lý.
