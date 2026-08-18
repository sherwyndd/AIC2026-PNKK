# spec.md - Video Keyframe Object Detection Module (YOLOv11s)

## 1. Tổng quan
Module phát hiện vật thể (Object Detection) cho các keyframe **đã được trích xuất sẵn** từ video. 
Module sử dụng mô hình **YOLOv11s** (Small) để nhận diện các vật thể xuất hiện trong frame, sau đó xử lý và gom nhóm thành các chuỗi văn bản và tần suất xuất hiện nhằm phục vụ việc đánh chỉ mục (Indexing) cho Elasticsearch trong bài toán Video Retrieval.

## 2. Luồng xử lý (Workflow)
1. **Data Loading:** Đọc danh sách các ảnh Keyframe (`.jpg`, `.png`) đã được trích xuất sẵn.
2. **YOLOv11s Inference:** Đưa các ảnh vào mô hình `yolov11s.pt` với ngưỡng tin cậy (`conf_threshold = 0.25`) để lọc dự đoán nhiễu.
3. **Data Aggregation:** Trích xuất danh sách nhãn (labels), chuyển đổi thành:
   * Chuỗi nhãn dạng chuỗi cách nhau bởi khoảng trắng (`detected_objects`).
   * Danh sách nhãn không trùng lặp (`unique_objects`).
   * Bảng đếm số lượng từng loại vật thể (`object_counts`).
4. **Export Data:** Lưu kết quả dạng mảng JSON chứa metadata của từng keyframe.

## 3. Cấu trúc lưu trữ JSON (Output Schema)
Mỗi đối tượng trong file JSON đầu ra chứa thông tin định danh `keyframe_id` cùng với dữ liệu vật thể được tổng hợp theo đúng định dạng tiêu chuẩn.

```json
[
  {
    "keyframe_id": "L01_V001_00450",
    "detected_objects": "car car person dog traffic_light",
    "unique_objects": [
      "car",
      "person",
      "dog",
      "traffic_light"
    ],
    "object_counts": {
      "car": 2,
      "person": 1,
      "dog": 1,
      "traffic_light": 1
    }
  },
  {
    "keyframe_id": "L01_V001_00480",
    "detected_objects": "",
    "unique_objects": [],
    "object_counts": {}
  }
]