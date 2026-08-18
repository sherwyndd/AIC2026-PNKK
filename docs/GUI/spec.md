# Specification: Keyframe & Metadata Inspector GUI (Demo Viewer)

## 1. Mục tiêu (Goal)
Tạo một Web GUI đơn giản, trực quan bằng **Streamlit** (hoặc **Gradio**) để kiểm tra kết quả sau khi chạy Pipeline cắt Shot và trích xuất Keyframes. Giúp người dùng dễ dàng kiểm tra tính chính xác của ranh giới Shot, vị trí Keyframe trên video gốc và dữ liệu Metadata đi kèm.

## 2. Công nghệ đề xuất (Tech Stack)
- **Framework UI:** `streamlit` (khuyên dùng cho layout dạng Dashboard) hoặc `gradio`.
- **Xử lý dữ liệu:** `pandas`.
- **Đọc ảnh / Video:** `PIL` (Pillow) hoặc `opencv-python`.

## 3. Kiến trúc giao diện & Trải nghiệm người dùng (UI/UX Design)

### 3.1. Sidebar (Cột bên trái - Control Panel)
- **Dropdown - Chọn Video:** Danh sách các `video_id` quét được từ thư mục `Metadata/videos/videos_metadata.csv`.
- **Card - Thông tin Video gốc:**
  - File name | FPS | Duration | Total Frames.
- **Selectbox - Lọc theo Shot:** `All Shots` hoặc chọn cụ thể `Shot 001`, `Shot 002`...
- **Slider - Kích thước ảnh Grid:** Cho phép chỉnh zoom ảnh Keyframe to/nhỏ để dễ soi.

### 3.2. Main Panel (Khu vực hiển thị chính)

#### Tab 1: Keyframe Viewer (Chính)
- **Video Player Integration:** Nằm ở trên cùng, hỗ trợ phát trực tiếp file `.mp4` tương ứng.
- **Keyframe Grid (Lưới ảnh đại diện):**
  - Hiển thị danh sách ảnh từ thư mục `Keyframes/[video_id]/`.
  - Nếu chọn Shot cụ thể: Chỉ lọc ra các Keyframe thuộc `shot_id` đó.
  - **Mỗi thẻ ảnh (Card) bao gồm:**
    - Ảnh Keyframe Thumbnail.
    - Nhãn bên dưới: `ID`, `Frame Index`, `Timestamp (s)` (VD: `3.0s | Frame 90`).
    - **Nút bấm "Jump to Video":** Bấm vào sẽ tua Video Player lên phía trên tới đúng giây (`pts_time`) của Keyframe đó.

#### Tab 2: Metadata Explorer (Soi dữ liệu thô)
- Hiển thị 2 bảng tương tác (Dataframe) có tính năng tìm kiếm / sắp xếp:
  1. **Shot Table (`[video_id]_shots.csv`):** Danh sách các Shot với `start_frame`, `end_frame`, `pts_start`, `pts_end`.
  2. **Keyframe Mapping Table (`[video_id]_keyframes.csv`):** Bảng chi tiết mapping giữa Keyframe và vị trí thời gian.

## 4. Luồng hoạt động của Demo Script (`app_demo.py`)

1. **Khởi tạo:**
   - Tự động kiểm tra sự tồn tại của thư mục `--output_dir` (chứa `Metadata/` và `Keyframes/`).
   - Đọc file tổng `Metadata/videos/videos_metadata.csv` để load danh sách Video.
2. **Xử lý Tương tác (Event Handling):**
   - Khi người dùng chọn 1 Video trên Sidebar $\rightarrow$ Đọc file `_shots.csv` và `_keyframes.csv` tương ứng.
   - Khi người dùng chọn 1 Shot $\rightarrow$ Filter danh sách Keyframe hiển thị trong Grid.
   - Khi người dùng click chọn 1 Keyframe $\rightarrow$ Cập nhật mốc thời gian phát của `st.video(..., start_time=pts_time)`.

## 5. Hướng dẫn chạy Demo (CLI Command)

Chạy file giao diện demo đơn giản qua dòng lệnh:

```bash
# Nếu dùng Streamlit:
streamlit run app_demo.py -- --data_dir ./output_dir

# Nếu dùng Gradio:
python app_demo.py --data_dir ./output_dir