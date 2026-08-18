# Pipeline Architecture

## Luồng đã triển khai

```text
Video input (/mlcv2025/... hoặc --input_dir)
   │
   ▼
[01_shot_cut.py] TransNetV2 → shots metadata
   │
   ▼
[02_keyframe_ext.py] every-8th sample → CLIP → L2 rel_diff > 0.4
   │
   ▼
Keyframes (.jpg) + metadata (csv/json)
```

## Output theo mode

| Mode | Output root | Metadata folder | Keyframes folder |
|------|-------------|-----------------|------------------|
| `demo` | `dataset/demo/` | `Metadata/` | `Keyframes/` |
| `full` | `data/` | `metadata/` | `keyframes/` |

## Entrypoints

| Script | Vai trò |
|--------|---------|
| `run_pipeline.py` | Chạy toàn bộ pipeline hoặc stage riêng |
| `src/preprocessing/01_shot_cut.py` | Chỉ cắt shot |
| `src/preprocessing/02_keyframe_ext.py` | Chỉ trích keyframe (cần shots metadata) |
| `src/preprocessing/03_asr_whisper.py` | Chỉ trích ASR và gán vào keyframe |
| `src/preprocessing/05_obj_detect.py` | Chỉ phát hiện vật thể trên keyframe |
| `src/preprocessing/06_captioning.py` | Chỉ sinh caption cho keyframe |
| `src/demo/demo_app.py` | GUI xem demo output |

## Blueprint đầy đủ

Sơ đồ folder toàn dự án (ASR, OCR, Milvus, search…) xem `pipeline_config.yaml` ở root.

Chi tiết kỹ thuật:
- `docs/shot + keyframes/spec.md`
- `docs/keyframes + filterv2/spec.md`

## Config

- `configs/runtime.yaml` — paths, mode, CLIP threshold, batch size
- `configs/search_config.yaml` — placeholder (chưa dùng)
