# Technical Specification: Two-Stage Keyframe Extraction Pipeline

## 1. Overview
Tài liệu này định nghĩa thiết kế kỹ thuật hoàn chỉnh cho hệ thống trích xuất và tinh lọc Keyframe hai lớp (**Two-Stage Keyframe Extraction Pipeline**). Hệ thống phục vụ bài toán Video Retrieval quy mô lớn bằng cách kết hợp **TransNetV2** (Phân đoạn cảnh vật lý) và **CLIP ViT-L/14-quickgelu** (Lọc trùng lặp ngữ nghĩa theo chuẩn $L_2$-Norm), nhằm tối ưu hóa số lượng keyframe và loại bỏ các khung hình dư thừa trước khi đưa vào CSDL Vector (Indexing).

---

## 2. Pipeline Architecture
```text
Raw Video Input (.mp4, .mkv, ...)
   │
   ▼
[ Stage 1: TransNetV2 ] ───────► Physical Shot Boundaries
   │
   ▼
[ Every-8th Frame Sampler ] ──► Candidate Frames Subsets
   │
   ▼
[ Stage 2: CLIP Embedder ] ────► Feature Embeddings (D = 768)
   │
   ▼
[ L2-Norm Pruning (rel_diff > 0.4) ]
   │
   ▼
Final Keyframe Dataset (.webp / .jpg) ──► Vector DB (Milvus / FAISS)
```

---

## 3. Detailed Stage Specifications

### Stage 1: Shot Boundary Detection (TransNetV2)
- **Objective:** Phát hiện chính xác các điểm chuyển cảnh (Hard cuts & Gradual transitions) để chia video thô thành các phân cảnh độc lập (Shots).
- **Input:** Video thô (`.mp4`, `.mkv`...).
- **Model Backbone:** TransNetV2 (3D-CNN architecture).
- **Output:** Danh sách các khoảng Shot $[F_{\text{start}}, F_{\text{end}}]$.

### Stage 2: Candidate Frame Sampling & Feature Extraction
- **Sampling Strategy (Lấy mẫu thô):**
  - Trong mỗi Shot thu được từ Stage 1, thực hiện lấy mẫu cố định với bước nhảy 8 frames/lần (every 8th frame).
  - **Mục tiêu:** Giảm $87.5\%$ lượng frame cần xử lý ở mô hình Deep Learning phía sau.
- **Feature Extraction (Trích xuất đặc trưng):**
  - **Model Backbone:** CLIP ViT-L/14-quickgelu (Pre-trained on DFN2B).
  - **Embedding Dimension:** $D = 768$.
  - **Output:** Mỗi candidate frame $i$ biến đổi thành vector $\mathbf{e}_i \in \mathbb{R}^{768}$.

### Stage 2 (Cont.): L2-Norm Semantic Pruning (Lọc Ngữ Nghĩa)
So sánh tuần tự từng candidate frame với Keyframe gần nhất được giữ lại trong cùng một Shot.

- **Formula (Độ chênh lệch tương đối):**
  $$\text{rel\_diff} = \frac{\Vert{}\mathbf{e}_{\text{current}} - \mathbf{e}_{\text{prev}}\Vert{}_2}{\Vert{}\mathbf{e}_{\text{prev}}\Vert{}_2}$$
  *Trong đó:*
  - $\mathbf{e}_{\text{current}}$: Vector của candidate frame đang xét.
  - $\mathbf{e}_{\text{prev}}$: Vector của keyframe chuẩn vừa được chấp nhận gần nhất trong cùng shot.

- **Decision Rules:**
  1. **First Frame Rule (Bắt buộc):** Frame đầu tiên của mỗi Shot luôn luôn được chọn làm Keyframe chính thức đầu tiên ($\mathbf{e}_{\text{prev}}$ gốc).
  2. **Threshold Rule:** Candidate frame được giữ lại làm Keyframe mới $\iff \text{rel\_diff} > 0.4$.
  3. **Discard Rule:** Nếu $\text{rel\_diff} \le 0.4$, frame bị coi là dư thừa/ảnh tĩnh và bị loại bỏ.
  4. **State Isolation:** Biến $\mathbf{e}_{\text{prev}}$ được reset hoàn toàn khi chuyển sang Shot tiếp theo.

---

## 4. End-to-End Pseudocode Implementation

```python
import cv2
import numpy as np
import torch

class TwoStageKeyframeExtractor:
    def __init__(self, transnet_model, clip_embedder, rel_diff_threshold=0.4, sample_step=8):
        self.transnet = transnet_model
        self.clip = clip_embedder
        self.threshold = rel_diff_threshold
        self.sample_step = sample_step

    def process_video(self, video_path):
        """
        Giai đoạn 1: Cắt Shot bằng TransNetV2
        """
        # shots là danh sách các tuple [(start_frame, end_frame), ...]
        shots = self.transnet.predict_shots(video_path) 
        final_keyframes = []

        cap = cv2.VideoCapture(video_path)
        
        for shot_idx, (start_frame, end_frame) in enumerate(shots):
            # Lấy mẫu candidate frames (Every 8th frame)
            candidate_indices = list(range(start_frame, end_frame + 1, self.sample_step))
            if not candidate_indices:
                continue

            candidate_frames = self._extract_frames_by_indices(cap, candidate_indices)
            
            # Giai đoạn 2: Lọc Semantic với CLIP & L2-Norm
            selected_frames = self._filter_stage_2(candidate_frames)
            final_keyframes.extend(selected_frames)

        cap.release()
        return final_keyframes

    def _filter_stage_2(self, candidate_frames):
        if not candidate_frames:
            return []

        # 1. Trích xuất Embeddings qua CLIP ViT-L/14-quickgelu (Pre-trained DFN2B)
        embeddings = self.clip.encode_images(candidate_frames) # Shape: (N, 768)
        
        selected_keyframes = []
        
        # Rule 1: Luôn giữ frame đầu tiên của Shot
        e_prev = embeddings[0]
        selected_keyframes.append(candidate_frames[0])

        # Rule 2 & 3: Lọc bằng L2-Norm Relative Difference
        for idx in range(1, len(candidate_frames)):
            e_current = embeddings[idx]
            
            # Tính rel_diff theo Euclidean Norm
            rel_diff = np.linalg.norm(e_current - e_prev) / np.linalg.norm(e_prev)

            if rel_diff > self.threshold:
                selected_keyframes.append(candidate_frames[idx])
                e_prev = e_current # Cập nhật mốc so sánh mới

        return selected_keyframes

    def _extract_frames_by_indices(self, cap, indices):
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        return frames
```

---

## 5. Runtime Configuration (Implementation)

Pipeline được tích hợp trong `run_pipeline.py` và `src/preprocessing/02_keyframe_ext.py` (Stage 2 CLIP + L2 pruning).

### 5.1 Defaults (chạy `python run_pipeline.py` không tham số)
| Tham số | Mặc định | Ghi chú |
|---------|----------|---------|
| `--input_dir` | `/mlcv2025/Datasets/HCMAI25/batch2/video` | Theo `docs/shot + keyframes/spec.md` |
| `--output_dir` | `dataset/demo` (demo) / `data` (full) | `Metadata/`+`Keyframes/` hoặc `metadata/`+`keyframes/` |
| `--mode` | `demo` | 1 video đầu tiên |
| `--format` | `csv` | Metadata |
| `--sample_step` | `8` | Candidate every-8th frame trong shot |
| `--rel_diff_threshold` | `0.4` | Ngưỡng L2 relative diff |
| `--clip_batch_size` | `8` | Batch encode CLIP |

### 5.2 CLIP Model (triển khai)
- **Library:** `transformers` + `Pillow`
- **Checkpoint:** `openai/clip-vit-large-patch14` (768-d)
- **Venv:** `/AIClub_NAS/core_baotg/phong/AIC_2026/.venv`

### 5.3 Output
- Chỉ lưu **keyframes sau lọc** (`.jpg`) và metadata schema cũ (`keyframe_id`, `shot_id`, `frame_idx`, `pts_time`, `image_path`, ...).
- **Không** lưu embedding ra disk trong phase này (Milvus/FAISS để sau).

### 5.4 Ví dụ chạy
```bash
source /AIClub_NAS/core_baotg/phong/AIC_2026/.venv/bin/activate
python run_pipeline.py --mode demo
```