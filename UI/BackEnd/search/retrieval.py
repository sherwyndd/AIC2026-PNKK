import logging
from typing import List, Dict, Any

# 1. Import các mô hình AI
from src.models.translator import QueryTranslator
from src.models.encoders import TextEncoder, ImageEncoder
from src.models.cross_encoder import CrossEncoderReranker
from src.models.vqa import QwenVQAModel

# 2. Import các module Database & Logic
from src.database.elasticsearch import search_asr, search_ocr, search_object
from src.database.qdrant import query_collection, COLLECTION_SIGLIP, COLLECTION_PECORE
from src.utils.fusion import reciprocal_rank_fusion, rrf_fuse
from src.utils.keyframe_manager import get_keyframe_timestamp

logger = logging.getLogger(__name__)

class VideoRetrievalPipeline:
    """
    Main Pipeline điều phối toàn bộ quá trình tìm kiếm Video.
    Text Query -> Translate -> Encode -> Search (Qdrant + ES) -> Fusion (RRF) -> Output
    """
    
    def __init__(self, config: Any):
        self.config = config
        logger.info("Đang khởi tạo VideoRetrievalPipeline...")
        
        # Khởi tạo các AI Models (Sử dụng cấu hình từ pydantic-settings)
        self.translator = QueryTranslator()
        
        # Giả định config của bạn có chứa tên model, nếu không bạn có thể hardcode tên vào đây
        text_model_name = getattr(self.config, 'text_encoder_model', "ViT-B-32")
        self.text_encoder = TextEncoder(model_name=text_model_name)
        
        # Nếu cấu hình có yêu cầu khởi tạo ImageEncoder (cho SigLIP/PECORE) thì bỏ comment
        # siglip_model_name = getattr(self.config, 'siglip_model', "google/siglip-base-patch16-224")
        # self.siglip_encoder = ImageEncoder(model_name=siglip_model_name)
        
        logger.info("✅ Pipeline đã sẵn sàng nhận Query!")

    def search(self, query: str, top_k: int = 60, video_list: List[str] = None) -> List[Dict]:
        """
        Thực thi tìm kiếm trên nhiều không gian (Semantic + Lexical) và gộp kết quả.
        """
        logger.info(f"Bắt đầu xử lý query: '{query}'")
        
        # Bước 1: Dịch Tiếng Việt -> Tiếng Anh
        eng_query = self.translator.translate(query)
        logger.info(f"Đã dịch: '{eng_query}'")
        
        # Bước 2: Chuyển text thành Vector nhúng (Embedding)
        # Lưu ý: Hàm encode_text cần trả về numpy array
        text_embedding = self.text_encoder.encode_text(eng_query)
        
        # Bước 3: Tìm kiếm ngữ nghĩa (Semantic Search) qua Qdrant
        # Ta dùng SigLIP SQ8 như trong cấu hình mặc định bạn đã cung cấp
        siglip_points = query_collection(
            embedding=text_embedding,
            collection_name=COLLECTION_SIGLIP,
            limit=top_k,
            video_list=video_list
        )
        
        # Format lại kết quả vector search để khớp chuẩn Dict đầu vào của hàm Fusion
        semantic_results = []
        for pt in siglip_points:
            # Qdrant trả về object ScoredPoint, dữ liệu lưu trong thuộc tính payload
            semantic_results.append({
                "video_id": pt.payload.get("video_id", ""),
                "keyframe_id": pt.payload.get("keyframe_id", ""),
                "frame_idx": pt.payload.get("frame_idx", 0),
                "image_path": pt.payload.get("image_path", ""),
                "shot_id": pt.payload.get("shot_id", ""),
                "score": pt.score
            })
            
        # Bước 4: Tìm kiếm từ khóa (Lexical Search) qua Elasticsearch
        asr_results = search_asr(query=eng_query, limit=top_k, video_list=video_list)
        ocr_results = search_ocr(query=eng_query, limit=top_k, video_list=video_list)
        object_results = search_object(query=eng_query, limit=top_k, video_list=video_list)
        
        # Bước 5: Kết hợp kết quả (Reciprocal Rank Fusion - RRF)
        results_by_source = {
            "semantic": semantic_results,
            "asr": asr_results,
            "ocr": ocr_results,
            "object": object_results
        }
        
        # Hàm fusion sẽ tự động tính điểm theo tệp trọng số mặc định (DEFAULT_SOURCE_WEIGHTS)
        fused_results = reciprocal_rank_fusion(
            results_by_source=results_by_source,
            top_k=top_k,
            k=60
        )
        
        # Bước 6: Nội suy Timestamp siêu tốc bằng Keyframe Manager
        # Phục vụ cho việc format kết quả chuẩn DRES
        for res in fused_results:
            vid = res.get("video_id")
            kfid = res.get("keyframe_id")
            fidx = res.get("frame_idx")
            
            # Gọi hàm ultra-fast lookup
            timestamp_ms = get_keyframe_timestamp(vid, kfid, fidx)
            res["timestamp"] = timestamp_ms
            
        logger.info(f"Hoàn thành xử lý, trả về {len(fused_results)} kết quả.")
        return fused_results