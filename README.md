# AI Challenge 2026 — Video Preprocessing Pipeline

Pipeline tiền xử lý video cho bài toán **Video Retrieval**: cắt shot (TransNetV2) → lấy candidate every-8th frame → lọc keyframe bằng CLIP L2-norm.

## Cấu trúc dự án

```text
AIC_2026/
├── configs/
│   ├── runtime.yaml          # paths & hyperparameters (đọc khi chạy)
│   └── search_config.yaml    # placeholder (search/rerank — chưa dùng)
├── pipeline_config.yaml      # sơ đồ cây blueprint toàn pipeline
├── data/                     # output FULL (mode full)
│   ├── keyframes/
│   └── metadata/
├── dataset/
│   └── demo/                 # output DEMO (mode demo)
├── src/
│   ├── preprocessing/
│   │   ├── 01_shot_cut.py
│   │   ├── 02_keyframe_ext.py
│   │   ├── 03_asr_whisper.py
│   │   ├── 05_obj_detect.py
│   │   └── 06_captioning.py
│   ├── pipeline/             # thư viện dùng chung
│   ├── demo/demo_app.py      # Streamlit inspector
│   └── utils/gpu_utils.py
├── run_pipeline.py           # master entrypoint
└── docs/
```

Các module chưa triển khai (`indexing/`, `search/`, `web_ui/`, ASR/OCR/…) giữ folder trống.

## Cài đặt

```bash
cd /AIClub_NAS/core_baotg/phong/AIC_2026
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy nhanh

### Demo (1 video → `dataset/demo/`)

```bash
source .venv/bin/activate
python run_pipeline.py --mode demo
```

### Full (toàn bộ video → `data/`)

```bash
python run_pipeline.py --mode full
```

### Batch runner — chạy stage riêng

```bash
source .venv/bin/activate

# Xóa object metadata cũ nếu muốn rerun lại stage này
rm -rf dataset/demo/Metadata/objects data/metadata/objects

# Chạy stage object detection riêng bằng batch scheduler
python run_batch_pipeline.py \
  --input_dir /mlcv2025/Datasets/HCMAI25/batch2/video \
  --output_dir ./data \
  --video_list ./video_list.txt \
  --gpus auto \
  --gpu_count 1 \
  --batch_size 5 \
  --stage object_detection \
  --mode full \
  --format csv
```

`--gpus auto` sẽ tự chọn GPU có dung lượng rảnh lớn nhất; vì bài toán object detection ở đây chỉ cần 1 GPU, nên đặt `--gpu_count 1`.

Đối với `run_pipeline.py`, stage riêng cũng hỗ trợ:

```bash
python run_pipeline.py --mode full --stage object_detection --video_list ./video_list.txt
```

### Xem kết quả demo

```bash
streamlit run src/demo/demo_app.py -- --data_dir ./dataset/demo
```

Video gốc mặc định đọc từ `configs/runtime.yaml` → `/mlcv2025/Datasets/HCMAI25/batch2/video`.

## Output

| Mode | Thư mục | Layout |
|------|---------|--------|
| `demo` | `dataset/demo/` | `Metadata/`, `Keyframes/` |
| `full` | `data/` | `metadata/`, `keyframes/` |

Metadata: `videos/`, `shots/`, `keyframes/` (CSV hoặc JSON).

## Tài liệu

- [Kiến trúc pipeline](docs/01_pipeline_arch.md)
- [Setup](docs/02_setup_guide.md)
- [CLI cheatsheet](docs/04_cheatsheet.md)
- [Spec shot + keyframes](docs/shot%20+%20keyframes/spec.md)
- [Spec CLIP filter v2](docs/keyframes%20+%20filterv2/spec.md)
