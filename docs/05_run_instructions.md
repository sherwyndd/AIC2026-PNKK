# Hướng dẫn chạy từng phần code

> Cập nhật sau tái cấu trúc pipeline. Module nội bộ (`src/pipeline/*`, `src/utils/*`) không có CLI riêng.

## 1. Master pipeline — `run_pipeline.py`

Chạy shot cut + keyframe extraction liên tiếp.

```bash
source .venv/bin/activate
python run_pipeline.py --mode demo
python run_pipeline.py --mode full
```

Output:
- `demo` → `dataset/demo/` (`Metadata/`, `Keyframes/`)
- `full` → `data/` (`metadata/`, `keyframes/`)

Use `--input_dir` to override the default video source from `configs/runtime.yaml`.
Use `--output_dir` to write results into a custom output directory instead of the mode defaults.

```bash
python run_pipeline.py --mode full --input_dir /path/to/videos --output_dir ./my_output
```

A log file is written to `<output_dir>/pipeline.log`. If a whole stage has completed successfully, the same stage will be skipped entirely on the next run. If a stage fails, the log will record `stage_failed` and the next run will retry that failed stage. If a stage was interrupted before finishing, rerunning it will rerun the whole stage from the beginning instead of resuming partially.

Use `--force` to rerun a completed stage even when the log says it is done.

## 1.5. Batch runner — chạy stage riêng `object_detection`

Nếu bạn muốn chạy riêng stage object detection theo format của `run_batch_pipeline.py`, hãy xóa object metadata cũ trước khi rerun để tránh đọc file stale:

```bash
source .venv/bin/activate
rm -rf dataset/demo/Metadata/objects data/metadata/objects

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

`--gpus auto` sẽ chọn GPU có VRAM rảnh lớn nhất; với bài toán object detection này, dùng 1 GPU là đủ nên `--gpu_count 1`.

Nếu cần chạy stage object detection đơn lẻ bằng master pipeline, dùng:

```bash
python run_pipeline.py --mode full --stage object_detection --video_list ./video_list.txt
```

Lưu ý: `run_batch_pipeline.py` hỗ trợ `--stage object_detection` theo batch, nên cocok khi cần chạy trên nhiều video song song theo GPU.

## 2. Stage 1 — `src/preprocessing/01_shot_cut.py`

```bash
python src/preprocessing/01_shot_cut.py --mode demo
```

Output: `{output}/metadata/shots/` hoặc `{output}/Metadata/shots/` + `videos_metadata.*`

## 3. Stage 2 — `src/preprocessing/02_keyframe_ext.py`

```bash
python src/preprocessing/02_keyframe_ext.py --mode demo
```

Output: keyframes `.jpg` + `{video_id}_keyframes.*`

## 4. Stage 3 — `src/preprocessing/03_asr_whisper.py`

```bash
python src/preprocessing/03_asr_whisper.py --mode demo --asr_format json
python src/preprocessing/03_asr_whisper.py --mode full --asr_format jsonl --asr_language vi --asr_beam_size 5
```

Output: ASR metadata files theo từng video vào thư mục `<output>/<metadata_dir>/asr/`.

## 5. Stage 4 — `src/preprocessing/05_obj_detect.py`

Chạy trực tiếp stage object detection:

```bash
python src/preprocessing/05_obj_detect.py --mode demo
python src/preprocessing/05_obj_detect.py --mode full --yolov8_model_id yolov11s.pt --det_conf_threshold 0.25 --det_iou_threshold 0.45
```

Chạy theo format batch scheduler:

```bash
source .venv/bin/activate
rm -rf dataset/demo/Metadata/objects data/metadata/objects

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

Output: Object detection metadata files theo từng video vào thư mục `<output>/<metadata_dir>/objects/`.

Nếu muốn rerun stage này, nên xóa thư mục `objects` cũ trước để tránh giữ metadata stale từ lần chạy trước. Nếu object detection fails for any reason, the pipeline logs `[PIPELINE] stage=object_detection status=stage_failed`. On a rerun, only failed stages are reattempted while completed stages are skipped.

## 6. Stage 5 — `src/preprocessing/06_captioning.py`

Mặc định hiện tại dùng model nhẹ hơn để tránh OOM trên GPU có VRAM còn ít:

```bash
python src/preprocessing/06_captioning.py --mode demo
python src/preprocessing/06_captioning.py --mode full --caption_model_id OpenGVLab/InternVL2_5-2B
```

Output: Caption metadata files theo từng video vào thư mục `<output>/<metadata_dir>/captions/`.

Với model mới, mặc định dùng `OpenGVLab/InternVL2_5-2B` ở chế độ INT4 để giảm VRAM. Nếu free VRAM còn thấp hoặc máy quá giới hạn, nên giảm `batch_size` và ưu tiên chạy trên 1 GPU để tránh OOM khi load model và xử lý từng keyframe.

Nếu captioning fails vì OOM hoặc model không đủ bộ nhớ, hãy tăng `--min_free_mem` lên 6000 hoặc chạy trên GPU khác còn trống hơn. Nếu caption đã chạy xong, file output tương ứng sẽ được skip lại để tránh chạy lại vô ích.

Ghi log vào `<output_dir>/pipeline.log`. Nếu chạy lại cùng command trên cùng output directory, stage sẽ tự động skip video đã có ASR metadata trước đó để hỗ trợ resume.

## 5. Demo GUI — `src/demo/demo_app.py`

```bash
streamlit run src/demo/demo_app.py -- --data_dir ./dataset/demo
```

Mặc định `--video_dir` = input path trong `configs/runtime.yaml`.

## Config

Chỉnh mặc định tại `configs/runtime.yaml`. Blueprint folder đầy đủ: `pipeline_config.yaml`.
