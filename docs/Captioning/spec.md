# spec.md - Video Keyframe Captioning Module (InternVL 2.5)

## 1. Tổng quan
Module sinh mô tả tự động (Image Captioning) cho các keyframe **đã được trích xuất sẵn** từ video. 
Module sử dụng mô hình Multimodal Large Language Model **OpenGVLab/InternVL2_5-2B** với chế độ nén INT4 (BitsAndBytes) để giảm VRAM tiêu thụ, phù hợp với GPU 8GB+ khi cần xử lý ảnh keyframe. Kết quả được lưu trực tiếp ra file định dạng JSON nhằm phục vụ cho bài toán Video Retrieval / VQA.

## 2. Luồng xử lý (Workflow)
1. **Data Loading:** Đọc danh sách các ảnh Keyframe (.jpg, .png) đã được trích xuất và lưu trữ sẵn trong thư mục theo cấu trúc của hệ thống.
2. **InternVL Inference:** Đưa từng ảnh vào mô hình `OpenGVLab/InternVL2_5-2B` kèm theo prompt chuẩn: *"Describe this image concisely in English, focusing on key objects, actions, text, and context."* để sinh mô tả.
3. **Export Data:** Tổng hợp kết quả ánh xạ giữa `frame_id` và `caption_en`, sau đó ghi toàn bộ ra file `.json`.

## 3. Cấu trúc lưu trữ JSON (Output Schema)
Mỗi đối tượng trong file JSON đầu ra sẽ lưu trữ thông tin của 1 Keyframe đi kèm mô tả từ mô hình.

```json
{
  "L01_V001_00450.jpg": {
    "model_id": "OpenGVLab/InternVL2_5-2B",
    "prompt_used": "Describe this image concisely in English...",
    "caption_en": "A local government official giving a speech at an outdoor opening ceremony with a blue backdrop.",
    "has_caption": true
  },
  "L01_V001_00480.jpg": {
    "model_id": "OpenGVLab/InternVL2_5-2B",
    "prompt_used": "Describe this image concisely in English...",
    "caption_en": "A wide shot of a crowd clapping their hands.",
    "has_caption": true
  }
}