# 📜 THÔNG TIN & LUẬT THI VÒNG SƠ TUYỂN AIC 2026

## 1. NỘI DUNG CÁC TRUY VẤN VÒNG SƠ TUYỂN

### 1.1. Truy vấn dạng 1: Tìm kiếm chính xác theo văn bản (Textual KIS)
*   **Mô tả:** Tìm kiếm sự kiện dựa trên mô tả bằng ngôn ngữ tự nhiên.
*   **Yêu cầu nộp:** Định vị chính xác đoạn video bằng cách chỉ ra một khung hình bất kỳ thuộc đoạn video đó. 
*   **Định dạng:** `video_id = video_abc(.mp4), frame_id = 1500`.

### 1.2. Truy vấn dạng 2: Truy vấn dạng Hỏi–Đáp (Q&A)
*   **Mô tả:** Tìm kiếm sự kiện và trích xuất thông tin cụ thể từ video dựa trên câu hỏi.
*   **Yêu cầu nộp:** Tìm ra chính xác khoảnh khắc liên quan và trả lời câu hỏi (tiếng Việt hoặc Anh).
*   **Định dạng:** `video_id = video_xyz(.mp4), frame_id = 3450, answer = "5"`.

### 1.3. Truy vấn dạng 3: Truy xuất và căn chỉnh sự kiện video theo thời gian (TRAKE)
*   **Mô tả:** Đòi hỏi độ chính xác cao trong truy xuất và căn chỉnh thời gian của một chuỗi sự kiện. Gồm 2 giai đoạn:
    1.  **Retrieval:** Tìm ra 1 video duy nhất chứa chuỗi sự kiện.
    2.  **Alignment:** Xác định chính xác 1 khung hình (semantic keyframe) duy nhất cho mỗi giai đoạn của chuỗi sự kiện.
*   **Khung hình ngữ nghĩa:** Khoảnh khắc mang ý nghĩa về nội dung (ví dụ: khoảnh khắc chân chạm đất), không phải I-Frame kỹ thuật.

---

## 2. PHƯƠNG PHÁP ĐÁNH GIÁ (SCORING)

Đội thi nộp tối đa **100 câu trả lời** cho mỗi truy vấn. Điểm cuối cùng là trung bình của những câu trả lời tốt nhất ở các vị trí xếp hạng khác nhau.

### 2.1. Điểm Tương Quan (R-Score) - Thang điểm 0 đến 1

*   **Truy vấn KIS:** `R-Score = 1` nếu đúng video VÀ frame nộp nằm trong khoảng đáp án `[s, e]`.
*   **Truy vấn Q&A:** `R-Score = 1` nếu đúng video, đúng frame `[s, e]`, VÀ answer khớp ý nghĩa.
*   **Truy vấn TRAKE:** `R-Score` tính bằng tỉ lệ khung hình khớp đáp án trên tổng số khoảnh khắc (Ví dụ khớp 3/4 sự kiện -> `0.75`). Nếu sai video -> `0`. Khoảng đáp án của TRAKE thường rất hẹp (< 10 frame).

### 2.2. Điểm Cuối Cùng (Final Score)

Được tính bằng trung bình cộng của **Top-k R-Score (R@k)** tại 5 mốc: $k \in \{1, 5, 20, 50, 100\}$.
*   $R@k$ = Điểm R-Score cao nhất trong k câu trả lời đầu tiên.
*   $\text{Final Score} = \frac{R@1 + R@5 + R@20 + R@50 + R@100}{5}$
*   **Mục đích:** Khuyến khích thí sinh đẩy đáp án đúng lên các vị trí xếp hạng cao nhất có thể (đặc biệt là Top 1).

---

## 3. THÔNG TIN DỮ LIỆU VÒNG SƠ TUYỂN – ĐỢT 1 (Batch 1)

*   **Videos:** Dữ liệu gốc để thi.
*   **Keyframes:** Đã trích xuất sẵn, lưu theo cấu trúc thư mục, số thứ tự tăng dần tương ứng metadata.
*   **Objects:** File JSON nhận diện vật thể bằng Faster R-CNN (OpenImages V4).
*   **CLIP features:** Vector nhúng từ mô hình `clip-ViT-B-32` (Lưu chung file .npy).
*   **Metadata:** Thông tin JSON từ YouTube.
*   *(Lưu ý: Dữ liệu thi chính thức chỉ là Video, các dữ liệu khác chỉ để tham khảo)*. Đợt 2 (Batch 2) sẽ được công bố sau.
