# CLI Cheatsheet

## Master pipeline

```bash
# Demo → dataset/demo/
python run_pipeline.py --mode demo

# Full → data/
python run_pipeline.py --mode full

# Chỉ stage 1 hoặc 2
python run_pipeline.py --mode demo --stage shot_cut
python run_pipeline.py --mode demo --stage keyframe_ext

# Ghi đè config
python run_pipeline.py --config configs/runtime.yaml --input_dir /path/to/videos --format json
```

## Stage 1 — Shot cut

```bash
python src/preprocessing/01_shot_cut.py --mode demo
python src/preprocessing/01_shot_cut.py --mode full --output_dir ./data --format csv
```

## Stage 2 — Keyframe extraction

```bash
python src/preprocessing/02_keyframe_ext.py --mode demo
python src/preprocessing/02_keyframe_ext.py --mode full --sample_step 8 --rel_diff_threshold 0.4
```

Cần shots metadata từ stage 1 trong cùng `output_dir`.

## Stage 3 — ASR Whisper

```bash
python src/preprocessing/03_asr_whisper.py --mode demo --asr_format json
python src/preprocessing/03_asr_whisper.py --mode full --asr_format jsonl --asr_language vi --asr_beam_size 5
```

Output ASR sẽ nằm trong `metadata/asr/` theo `mode`.
Rerun the same command on the same output directory to resume processing. Already completed videos are skipped automatically based on existing ASR metadata files, and progress is logged to `<output_dir>/pipeline.log`.

## Stage 4 — Object detection

```bash
python src/preprocessing/05_obj_detect.py --mode demo
python src/preprocessing/05_obj_detect.py --mode full --yolov8_model_id yolov11s.pt --det_conf_threshold 0.25 --det_iou_threshold 0.45
```

Output object metadata sẽ nằm trong `metadata/objects/` theo `mode`.

## Stage 5 — Captioning

```bash
python src/preprocessing/06_captioning.py --mode demo
python src/preprocessing/06_captioning.py --mode full --caption_model_id OpenGVLab/InternVL2_5-2B
```

Output caption metadata sẽ nằm trong `metadata/captions/` theo `mode`.

## Demo app

```bash
streamlit run src/demo/demo_app.py -- --data_dir ./dataset/demo
streamlit run src/demo/demo_app.py -- --data_dir ./dataset/demo --video_dir /mlcv2025/Datasets/HCMAI25/batch2/video
```

## Tham số thường dùng

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `--mode` | `demo` | `demo`: 1 video; `full`: tất cả |
| `--format` | `csv` | `csv` hoặc `json` |
| `--sample_step` | `8` | Bước lấy candidate frame |
| `--rel_diff_threshold` | `0.4` | Ngưỡng CLIP L2 |
| `--clip_batch_size` | `8` | Batch encode CLIP |
