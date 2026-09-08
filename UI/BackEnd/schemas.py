"""Pydantic models for request/response validation."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SearchQueryItem(BaseModel):
    type: str = Field(..., description="Loại tìm kiếm: 'text' | 'ocr' | 'asr' | 'object' | 'image' | 'keyframe'")
    text: Optional[str] = Field(default=None, description="Chuỗi tìm kiếm cho text/ocr/asr/object")
    query: Optional[str] = Field(default=None, description="Alias cho text")
    video_id: Optional[str] = Field(default=None, description="Video ID cho tìm kiếm bằng Keyframe (Image Similarity)")
    frame_id: Optional[str] = Field(default=None, description="Frame ID / Frame Index cho tìm kiếm bằng Keyframe")
    frame_idx: Optional[int] = Field(default=None, description="Frame Index")
    keyframe_id: Optional[str] = Field(default=None, description="Keyframe ID")


class SearchRequest(BaseModel):
    search_queries: Optional[List[SearchQueryItem]] = Field(
        default=None,
        description="Danh sách các truy vấn tìm kiếm linh hoạt (text, ocr, asr, image similarity)",
    )
    query: Optional[str] = Field(default=None, description="Text query mô tả nội dung cần tìm (tương thích ngược)")
    ocr_query: Optional[str] = Field(default=None, description="Từ khóa OCR cần tìm")
    asr_query: Optional[str] = Field(default=None, description="Từ khóa ASR cần tìm")
    semantic_query: Optional[str] = Field(default=None, description="Text query SigLIP2 semantic")
    object_query: Optional[str] = Field(
        default=None,
        description='Object query, e.g. "3 person 2 car" (matched on ES object_text)',
    )
    top_k: int = Field(..., gt=0, description="Số ảnh trả về sau RRF fusion")
    rrf_k: int = Field(default=60, ge=1, description="Hệ số k của RRF (từ Settings React)")
    video_list: Optional[List[str]] = Field(
        default=None,
        description="Danh sách video_id cần lọc. Rỗng/None = tìm toàn bộ DB",
    )
    use_ocr: bool = Field(default=False, description="Bật chế độ OCR")
    use_asr: bool = Field(default=False, description="Bật chế độ ASR")
    use_reranker: bool = Field(default=True, description="Bật BGE-Reranker-v2-m3")
    reranker_top_n: int = Field(default=50, ge=5, description="Số lượng ứng viên mang đi rerank")

    # We skip the query non-empty validator because query is now optional
    # and we can search via ocr_query, asr_query or semantic_query instead.


class ImageResult(BaseModel):
    rank: int
    score: float
    video_id: str
    keyframe_id: str
    frame_idx: int
    image_url: str
    pts_time: Optional[float] = Field(default=None, description="Thời điểm theo giây")
    timestamp: Optional[int] = Field(default=None, description="Thời điểm theo milliseconds")
    matched_sources: Optional[List[str]] = Field(default=None, description="Danh sách các nguồn khớp kết quả này")


class KeyframeMeta(BaseModel):
    keyframe_id: str
    shot_id: Optional[str] = None
    video_id: str
    frame_idx: int
    pts_time: float
    timestamp: int
    image_path: str
    image_url: str
    asr_6s: Optional[str] = None
    asr_text: Optional[str] = None
    full_text: Optional[str] = None


class VideoKeyframesResponse(BaseModel):
    video_id: str
    total_keyframes: int
    fps: Optional[float] = Field(default=25.0, description="FPS của video từ videos_metadata.csv")
    keyframes: List[KeyframeMeta]


class SearchResponse(BaseModel):
    query: Optional[str] = None
    top_k: int
    rrf_k: int
    results: List[ImageResult]
    sources_used: Optional[List[str]] = Field(default=None, description="Danh sách các nguồn đã sử dụng tìm kiếm")


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    qdrant_connected: bool
