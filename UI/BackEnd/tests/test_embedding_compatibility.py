"""Test embedding compatibility between current model and Qdrant collection."""

import numpy as np
from qdrant_client import QdrantClient
from models.siglip_encoder import load_siglip, encode_text_siglip

# Load SigLIP2
print("Loading SigLIP2 model...")
load_siglip(device="cuda:0")

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

# Get a sample vector from collection
print("\nGetting sample vector from collection...")
sample_points = client.query_points(
    collection_name="image_siglip",
    query=[0.1] * 768,  # dummy query
    limit=1,
    with_payload=False
).points

if sample_points:
    sample_point_id = sample_points[0].id
    print(f"Sample point ID: {sample_point_id}")
    
    # Get the actual vector
    point_data = client.retrieve(
        collection_name="image_siglip",
        ids=[sample_point_id],
        with_vectors=True
    )
    
    if point_data:
        collection_vector = point_data[0].vector
        print(f"Collection vector shape: {len(collection_vector)}")
        print(f"Collection vector stats: min={min(collection_vector):.4f}, max={max(collection_vector):.4f}, mean={np.mean(collection_vector):.4f}")
        
        # Encode same text with current model
        test_query = "test"
        model_vector = encode_text_siglip(test_query)
        print(f"\nModel vector shape: {model_vector.shape}")
        print(f"Model vector stats: min={model_vector.min():.4f}, max={model_vector.max():.4f}, mean={model_vector.mean():.4f}")
        
        # Check similarity between random vectors (should be low)
        similarity = np.dot(collection_vector / np.linalg.norm(collection_vector), 
                           model_vector / np.linalg.norm(model_vector))
        print(f"\nRandom similarity (should be low ~0): {similarity:.4f}")
        
        # Test with same text encoding twice (should be identical)
        model_vector2 = encode_text_siglip(test_query)
        similarity_same = np.dot(model_vector / np.linalg.norm(model_vector), 
                                model_vector2 / np.linalg.norm(model_vector2))
        print(f"Same text similarity (should be 1.0): {similarity_same:.4f}")
