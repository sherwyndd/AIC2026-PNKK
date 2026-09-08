"""Test re-index script with small sample."""

import os
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from models.siglip_encoder import load_siglip

# Config
KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
COLLECTION_NAME = "image_siglip_test"
DEVICE = "cuda:0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image_siglip(image_path: str) -> np.ndarray:
    """Encode image with SigLIP."""
    from models.siglip_encoder import _siglip_model, _siglip_processor, _siglip_device
    
    image = Image.open(image_path).convert('RGB')
    inputs = _siglip_processor(images=image, return_tensors="pt").to(_siglip_device)
    
    import torch
    with torch.no_grad():
        image_features = _siglip_model.get_image_features(**inputs)
    
    if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
        image_features = image_features.pooler_output
    elif hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
        image_features = image_features.image_embeds
    elif hasattr(image_features, "last_hidden_state") and image_features.last_hidden_state is not None:
        image_features = image_features.last_hidden_state[:, 0, :]
    elif not isinstance(image_features, torch.Tensor):
        image_features = image_features[0]
    
    import torch.nn.functional as F
    image_emb = F.normalize(image_features, p=2, dim=-1)[0].cpu().float().numpy()
    return image_emb.astype(np.float32)

def main():
    # Load model
    logger.info("Loading SigLIP model...")
    load_siglip(device=DEVICE)
    
    # Initialize Qdrant client
    client = QdrantClient(host="localhost", port=6333)
    
    # Delete test collection if exists
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted old test collection")
    except:
        pass
    
    # Create test collection
    logger.info(f"Creating test collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.DOT)  # Use DOT for cosine similarity with normalized vectors
    )
    
    # Get sample keyframes (first 10 from first video)
    logger.info("Getting sample keyframes...")
    video_dir = Path(KEYFRAME_ROOT) / "L24_V005"
    if not video_dir.exists():
        video_dir = list(Path(KEYFRAME_ROOT).iterdir())[0]
    
    sample_images = list(video_dir.glob("*.jpg"))[:10]
    logger.info(f"Found {len(sample_images)} sample images")
    
    # Process sample
    points = []
    for i, img_path in enumerate(sample_images):
        try:
            rel_path = img_path.relative_to(KEYFRAME_ROOT)
            video_id = rel_path.parts[0]
            frame_idx = int(rel_path.stem.split("_kf_")[1])
            keyframe_id = rel_path.stem
            
            logger.info(f"Processing {i+1}/{len(sample_images)}: {img_path.name}")
            embedding = encode_image_siglip(str(img_path))
            logger.info(f"  Embedding shape: {embedding.shape}, stats: min={embedding.min():.4f}, max={embedding.max():.4f}")
            
            point = PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "video_id": video_id,
                    "keyframe_id": keyframe_id,
                    "frame_idx": frame_idx,
                    "image_path": str(rel_path)
                }
            )
            points.append(point)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
    
    # Upload points
    logger.info(f"Uploading {len(points)} points...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    # Test query
    logger.info("\nTesting query...")
    from models.siglip_encoder import encode_text_siglip
    test_query = "building"
    query_vec = encode_text_siglip(test_query).tolist()
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=5
    ).points
    
    logger.info(f"Query '{test_query}' results:")
    for i, point in enumerate(results):
        logger.info(f"  {i+1}. Score: {point.score:.4f}, Video: {point.payload.get('video_id')}, Frame: {point.payload.get('frame_idx')}")
    
    logger.info("\nTest completed successfully!")

if __name__ == "__main__":
    main()
