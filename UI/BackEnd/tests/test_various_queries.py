"""Test various queries to check if results are relevant."""

from qdrant_client import QdrantClient
from models.siglip_encoder import load_siglip, encode_text_siglip

# Load SigLIP2
print("Loading SigLIP2 model...")
load_siglip(device="cuda:0")

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

test_queries = [
    "building",
    "person",
    "car",
    "tree",
    "sky",
    "a tall building",  # from user screenshot
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    
    siglip_vec = encode_text_siglip(query).tolist()
    res_siglip = client.query_points(
        collection_name="image_siglip",
        query=siglip_vec,
        limit=5
    ).points
    
    print(f"Found {len(res_siglip)} results:")
    for i, point in enumerate(res_siglip):
        print(f"  {i+1}. Score: {point.score:.4f}, Video: {point.payload.get('video_id')}, Frame: {point.payload.get('frame_idx')}")
