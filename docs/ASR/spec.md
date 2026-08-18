# spec.md - ASR Extraction Module (TycheVid 6-Second Window)

## 1. Tổng quan
Module trích xuất văn bản từ giọng nói (Speech-to-Text) sử dụng `faster-whisper`.
Thay vì lưu chuỗi text chạy dọc theo audio, module này áp dụng kỹ thuật **Keyframe-Centered 6-Second Time Window** (chuẩn TycheVid) để gom nhóm câu thoại trực tiếp vào từng Keyframe, hỗ trợ lưu trữ vào JSON/Polars offline.

## 2. Luồng xử lý (Workflow)
1. **Audio Extraction:** Tách `.mp4` thành `.wav` (16kHz, mono) bằng FFmpeg.
2. **Transcription:** Chạy Whisper 1 lần trên file audio -> Thu được mảng các Segments (start_time, end_time, text).
3. **Window Mapping:** Với mỗi Keyframe ở mốc `T_kf`, gom toàn bộ text của các segments rơi vào khoảng `[T_kf - 3.0s, T_kf + 3.0s]`.
4. **Export Data:** Lưu kết quả ra định dạng JSON/JSONL.

## 3. Cấu trúc lưu trữ JSON (Metadata Schema)
Mỗi object JSON đại diện cho 1 Keyframe kèm theo dữ liệu ASR 6 giây bao quanh nó.

```json
{
  "video_id": "L01_V001",
  "shot_id": 12,
  "keyframe_id": "L01_V001_00450",
  "frame_idx": 450,
  "pts_time": 18.0,
  "asr_metadata": {
    "window_start": 15.0,
    "window_end": 21.0,
    "asr_text_6s": "chủ tịch tỉnh phát biểu tại buổi lễ khai mạc",
    "has_speech": true,
    "raw_segments": [
      {
        "segment_start": 14.5,
        "segment_end": 18.2,
        "text": "chủ tịch tỉnh phát biểu"
      },
      {
        "segment_start": 18.2,
        "segment_end": 21.5,
        "text": "tại buổi lễ khai mạc"
      }
    ]
  }
}
```

**Giải thích các trường:**
* `pts_time`: Thời gian chính xác của keyframe trong video (tính bằng giây).
* `asr_text_6s`: Nội dung ghép lại của tất cả segment khớp với cửa sổ 6 giây.
* `has_speech`: Cờ (boolean) đánh dấu đoạn này có tiếng người nói hay không để filter cho nhanh.
* `raw_segments`: Giữ lại mốc thời gian gốc của từng đoạn thoại con để tiện debug.

## 4. Script Python minh họa bằng Polars

```python
import polars as pl
from faster_whisper import WhisperModel

def extract_raw_asr(video_path: str, model: WhisperModel) -> list[dict]:
    """Transcribe video thành các segments gốc"""
    segments, _ = model.transcribe(video_path, language="vi", beam_size=5)
    return [
        {"start_time": s.start, "end_time": s.end, "text": s.text.strip()}
        for s in segments
    ]

def map_asr_to_keyframes_6s(keyframes_df: pl.DataFrame, asr_segments: list[dict]) -> pl.DataFrame:
    """Map ASR vào Keyframes dùng cửa sổ 6s [T_kf - 3, T_kf + 3]"""
    asr_df = pl.DataFrame(asr_segments)
    
    # Check nếu audio trống không có tiếng
    if asr_df.is_empty():
        return keyframes_df.with_columns(
            pl.lit("").alias("asr_text_6s"),
            pl.lit(False).alias("has_speech")
        )

    asr_results = []
    
    for row in keyframes_df.iter_rows(named=True):
        t_kf = row["pts_time"]
        w_start = max(0.0, t_kf - 3.0)
        w_end = t_kf + 3.0
        
        # Filter các segment giao với cửa sổ [w_start, w_end]
        matched_texts = asr_df.filter(
            (pl.col("end_time") > w_start) & (pl.col("start_time") < w_end)
        )["text"].to_list()
        
        asr_results.append({
            "keyframe_id": row["keyframe_id"],
            "asr_text_6s": " ".join(matched_texts),
            "has_speech": len(matched_texts) > 0
        })

    # Join metadata vào lại dataframe chính
    return keyframes_df.join(pl.DataFrame(asr_results), on="keyframe_id", how="left")
```