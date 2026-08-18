# Setup Guide

## 1. Virtual environment

```bash
cd /AIClub_NAS/core_baotg/phong/AIC_2026
source .venv/bin/activate   # hoặc: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Dependencies chính

- `torch`, `torchvision` — TransNetV2 + CLIP
- `transnetv2-pytorch` — shot boundary detection
- `transformers`, `accelerate`, `pillow` — CLIP ViT-L/14
- `opencv-python`, `pandas`, `tqdm`, `PyYAML`

## 3. Model weights

- **TransNetV2**: tự tải khi khởi tạo `TransNetV2(device=...)`
- **CLIP**: `openai/clip-vit-large-patch14` qua HuggingFace (lần chạy đầu cần network)

Cache có thể đặt trong `models/transnetv2/`, `models/clip/` (tùy chọn, chưa bắt buộc).

## 4. Input video

Mặc định trong `configs/runtime.yaml`:

```yaml
paths:
  input_dir: /mlcv2025/Datasets/HCMAI25/batch2/video
```

Ghi đè bằng `--input_dir` khi chạy pipeline.

## 5. GPU

Pipeline tự chọn `cuda` nếu có, ngược lại `cpu`. Sau mỗi stage gọi `free_vram()` dọn cache GPU.

## 6. Demo GUI

```bash
pip install streamlit
streamlit run src/demo/demo_app.py -- --data_dir ./dataset/demo
```
