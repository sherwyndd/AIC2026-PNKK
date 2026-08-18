# Video Shot Segmentation & Keyframe Extraction Pipeline

Kế hoạch thực thi đầy đủ để xây dựng pipeline tiền xử lý (Preprocessing) cho bài toán Video Retrieval theo cấu trúc `spec.md`.

## Proposed Changes

Pipeline sẽ bao gồm một file chính chạy kịch bản (script) và một số bước thiết lập môi trường để lấy model.

### 1. Khởi tạo và Thiết lập Môi trường (Environment Setup)
Tôi sẽ chạy các lệnh command-line để thực hiện việc này:
- Clone repo chính thức của TransNetV2 từ GitHub.
- Tải file `transnetv2-weights.pth` (có thể từ repo của tác giả hoặc Google Drive link do tác giả cung cấp) về để chung vào thư mục project.
- *Lưu ý:* Việc dùng model từ clone của TransNetV2 sẽ yêu cầu import đúng class theo file `.py` bên trong repo (ví dụ `transnetv2.py`).

### 2. Cấu trúc Dự án & Code
#### [MODIFY] [requirements.txt](file:///AIClub_NAS/core_baotg/phong/AIC_2026/requirements.txt)
Thêm các thư viện cần thiết:
```text
opencv-python
pandas
numpy
matplotlib
torch
torchvision
tqdm
```

#### [NEW] [main.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/main.py)
Script Python cốt lõi sẽ xử lý logic từ đầu đến cuối:
- **CLI Arguments**: `argparse` với mặc định theo `spec.md` (`DEFAULT_*` trong `main.py`). Chạy `python main.py` không cần tham số; `--input_dir`, `--output_dir`, `--mode` (full, demo), `--format` (csv, json) chỉ để ghi đè.
- **Init Model**: Tải file `transnetv2-weights.pth` vào mô hình. Setup thiết bị GPU/CPU.
- **Vòng lặp Video (Video Loop)**: 
  - Duyệt các file video `.mp4`, `.mkv` trong `--input_dir`.
  - Nếu `--mode demo`, chỉ xử lý video đầu tiên.
- **Probe Metadata**: Mở video bằng `cv2.VideoCapture`, đọc `fps`, `total_frames`, tính `duration`, `width`, `height`. Ghi vào danh sách Video Metadata.
- **Segmentation (TransNetV2)**:
  - Hàm dự đoán frame chuyển cảnh của TransNetV2 sẽ trả ra list các shots (mảng các mốc frame bắt đầu và kết thúc).
  - Khởi tạo danh sách Shot Metadata tương ứng với output này.
- **Keyframe Extraction (Trích xuất ảnh)**:
  - Vòng lặp qua từng shot (khoảng `[start_frame, end_frame]` inclusive).
  - Lấy các frame cách nhau **8** frame, bắt đầu từ `start_frame`: `start_frame + k * 8` (`k = 0, 1, 2, ...`) cho đến khi `<= end_frame` (trong shot: frame 1-based `1, 9, 17, ...`).
  - Dùng `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, ...)` để seek đến vị trí.
  - **Xử lý frame lỗi (Corrupted Frames)**:
    - Nếu không `read()` được target frame, thử `target_frame - 1`.
    - Nếu vẫn lỗi, thử `target_frame + 1`.
    - Nếu cả hai vẫn lỗi, ghi log (cảnh báo) và skip.
    - Tính thời gian `pts_time = frame_idx / fps`.
    - Ghi ảnh bằng `cv2.imwrite` vào thư mục `[output_dir]/Keyframes/[video_id]/...`
  - Ghi thông tin vào Keyframe Metadata.
- **Export Data**:
  - Convert các mảng metadata thành `pandas.DataFrame`.
  - Nếu `--format csv`: Lưu đuôi `.csv`.
  - Nếu `--format json`: Lưu dưới dạng dictionary rồi dump ra `.json` hoặc dùng `df.to_json(orient='records')` để xuất danh sách ra `.json`.
  - Đảm bảo lưu đúng đường dẫn `[output_dir]/Metadata/...` như spec yêu cầu.
- **Dọn dẹp**: Gọi `torch.cuda.empty_cache()` sau mỗi video.

## Verification Plan

### Automated Tests
1. Chạy lệnh cài đặt thư viện (`pip install -r requirements.txt`).
2. Chạy tải file weights và clone code của TransNetV2 (nếu cần thiết sẽ tạo file script bash để tải).
3. Chạy script: `python main.py` (dùng mặc định spec) hoặc `python main.py --mode full` khi cần ghi đè.
4. Chạy lại script với `--format json`.

### Manual Verification
- Kiểm tra terminal output xem có hiển thị thanh tiến trình bằng `tqdm` không.
- Kiểm tra thư mục output xem có sinh ra các file `.csv`, `.json` chuẩn xác như yêu cầu của `spec.md` không.
- Đối chiếu số liệu trong file (keyframe_id, frame_idx, pts_time) với độ dài video.
- Kiểm tra các ảnh sinh ra trong thư mục `Keyframes/`.
