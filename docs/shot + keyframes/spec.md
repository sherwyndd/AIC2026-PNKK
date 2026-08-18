# Specification: Video Shot Segmentation & Keyframe Extraction Pipeline

## 1. Bối cảnh bài toán (Problem Context)
Đây là pipeline tiền xử lý (Preprocessing) cho bài toán **Video Retrieval** (Tìm kiếm sự kiện trong video). 
- **Mục tiêu:** Chuyển đổi hàng nghìn giờ video thô thành các vector có thể tìm kiếm được. 
- **Quy trình:** Hệ thống không tìm kiếm trực tiếp trên video chạy liên tục, mà phải cắt video thành các cảnh nhỏ (Shots) dựa trên sự thay đổi ngữ nghĩa. Từ mỗi cảnh nhỏ này, rút trích ra các bức ảnh đại diện (Keyframes). Sau đó, các Keyframes này mới được đưa qua AI (như CLIP) để lấy vector lưu vào database (Milvus).
- **Vai trò của Metadata:** Khi người dùng query và database trả về một kết quả, hệ thống chỉ nhận được `keyframe_id`. Dựa vào bộ Metadata này, hệ thống phải truy ngược lại được: Bức ảnh đó nằm ở giây thứ mấy (`pts_time`), thuộc phân đoạn nào (`shot_id` với `start_time`, `end_time` để phát video cho người xem), và thuộc file video gốc nào (`video_name`).

## 2. Yêu cầu đầu vào (Inputs & Options)
- **Thư mục đầu vào (mặc định):** `/mlcv2025/Datasets/HCMAI25/batch2/video` — cấu hình trong `configs/runtime.yaml`, ghi đè bằng `--input_dir`.
- **Thư mục đầu ra:**
  - `demo` → `dataset/demo/` (`Metadata/`, `Keyframes/`)
  - `full` → `data/` (`metadata/`, `keyframes/`)
- **Chạy không tham số:** `python run_pipeline.py` (đọc `configs/runtime.yaml`).

| Tham số | Mặc định | Ghi chú |
|---------|----------|---------|
| `--input_dir` | `/mlcv2025/Datasets/HCMAI25/batch2/video` | Thư mục video gốc |
| `--output_dir` | `dataset/demo` (demo) / `data` (full) | Tự chọn theo `--mode` |
| `--mode` | `demo` | `full`: toàn bộ video; `demo`: 1 video đầu tiên |
| `--format` | `csv` | Metadata: `csv` hoặc `json` |

- **Entrypoint:** `run_pipeline.py` (hoặc `src/preprocessing/01_shot_cut.py`, `02_keyframe_ext.py` chạy từng stage).

## 3. Định nghĩa Metadata Schema (Yêu cầu BẮT BUỘC)

Hệ thống phải xuất ra 3 loại file Metadata tương ứng với 3 cấp độ thông tin, với các trường (cột) phải tuân thủ chính xác như sau:

### 3.1. Video Metadata (`videos_metadata.csv`)
Lưu **toàn bộ thông tin kỹ thuật** đọc được từ video (không giới hạn trường cố định). Các trường tối thiểu bắt buộc:
- `video_id` (string): Lấy từ tên file bỏ đuôi mở rộng.
- `file_name` (string): Tên file gốc.
- `fps` (float): Tốc độ khung hình.
- `total_frames` (int): Tổng số frame.
- `duration` (float): Thời lượng (giây).
- `width` (int): Chiều ngang (pixels).
- `height` (int): Chiều dọc (pixels).

Ngoài ra, trích xuất thêm **tất cả** các thuộc tính có giá trị hợp lệ mà OpenCV/ffprobe đọc được, ví dụ: `fourcc`, `codec`, `bitrate`, `aspect_ratio`, `frame_size_bytes`, `pos_msec`, `brightness`, `contrast`... Bỏ qua thuộc tính nào trả về 0 hoặc không hợp lệ.

### 3.2. Shot Metadata (`[video_id]_shots.csv`)
Lưu thông tin ranh giới của từng phân đoạn cảnh sau khi cắt bằng TransNetV2.
- `shot_id` (string): Định danh duy nhất của shot (VD: `L21_V001_shot_001`).
- `video_id` (string): Map với ID của video chứa nó.
- `start_frame` (int): Frame bắt đầu cảnh.
- `end_frame` (int): Frame kết thúc cảnh.
- `pts_start` (float): Thời gian bắt đầu cảnh tính bằng giây (frame / fps).
- `pts_end` (float): Thời gian kết thúc cảnh tính bằng giây.
- `duration_frames` (int): Độ dài shot tính bằng số frame.

### 3.3. Keyframe Metadata (`[video_id]_keyframes.csv`)
Bảng ánh xạ (Mapping) cực kỳ quan trọng dùng để truy xuất ngược từ hình ảnh ra video thời gian thực.
- `keyframe_id` (string): Định danh duy nhất — đặt theo quy tắc `<video_id>_kf_XXXX`.
- `shot_id` (string): Thuộc shot nào (để lấy ranh giới start/end khi playback).
- `video_id` (string): Thuộc video nào.
- `frame_idx` (int): Chỉ số frame gốc bốc ra từ video.
- `pts_time` (float): Thời điểm xuất hiện tính bằng giây.
- `image_path` (string): Đường dẫn tương đối tới file ảnh đã lưu.

## 4. Luồng xử lý kỹ thuật (Processing Logic)
1. **Init Environment & Model**:
    - Kích hoạt Python Virtual Environment tại đường dẫn: /AIClub_NAS/core_baotg/phong/AIC_2026/.venv.
    - Đảm bảo đã cài đặt đầy đủ các gói phụ thuộc (transnetv2-pytorch, torch, opencv-python, pandas, transformers, accelerate).
    - Load weights TransNetV2 và CLIP `openai/clip-vit-large-patch14`; tự động cấu hình thiết bị tính toán (cuda nếu có GPU, ngược lại dùng cpu).
2. **Video Loop:** Quét thư mục `--input_dir`, lấy từng file video. (Nếu `--mode demo` chỉ lấy 1 file).
3. **Probe:** Đọc thông số file ghi vào mảng `Video Metadata`.
4. **Segment:** Pass video qua `model.predict_video()` để lấy ranh giới các shots. Ghi vào mảng `Shot Metadata`.
5. **Extract Keyframes (Two-Stage — chi tiết Stage 2 xem `docs/keyframes + filterv2/spec.md`):**
   - Lặp qua từng shot (khoảng `[start_frame, end_frame]` inclusive).
   - **Sampling:** Lấy candidate every-8th frame: `frame_idx = start_frame + k * 8`, `k = 0, 1, 2, ...`, `frame_idx <= end_frame`.
   - **CLIP filter:** Embed candidate bằng `openai/clip-vit-large-patch14`; giữ frame đầu shot; các frame sau giữ khi `rel_diff > 0.4` (so với keyframe vừa chấp nhận gần nhất trong cùng shot).
   - Chỉ các frame **sau lọc** mới `.imwrite()` ra ổ cứng và ghi `Keyframe Metadata`.
6. **Export:** Dump các mảng metadata ra các file `.csv` tương ứng trong thư mục `--output_dir/Metadata/`.
7. **Cleanup:** Chạy `torch.cuda.empty_cache()` để dọn VRAM trước khi sang video kế tiếp.

## 5. Cấu trúc Output mong muốn
```text
[output_dir]/
├── Keyframes/
│   └── L21_V001/
│       ├── L21_V001_kf_0001.jpg
│       └── L21_V001_kf_0002.jpg
└── Metadata/
    ├── videos/
    │   └── videos_metadata.csv
    ├── shots/
    │   └── L21_V001_shots.csv
    └── keyframes/
        └── L21_V001_keyframes.csv