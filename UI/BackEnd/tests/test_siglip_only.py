"""Test SigLIP2 only (no Jina) to verify encoding and Qdrant search."""
from qdrant_client import QdrantClient
from models.siglip_encoder import load_siglip, encode_text_siglip

# Load SigLIP2
print("Loading SigLIP2 model...")
load_siglip(device="cuda:0")

# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)

text_query = "cô gái mặc áo đỏ đi xe máy"
top_k = 10

# Query SigLIP
print(f"\nEncoding query: '{text_query}'")
siglip_vec = encode_text_siglip(text_query).tolist()
print(f"Encoding done. Shape: {len(siglip_vec)}")

print(f"\nQuerying Qdrant collection 'image_siglip'...")
res_siglip = client.query_points(
    collection_name="image_siglip",
    query=siglip_vec,
    limit=top_k
).points

print(f"\nSigLIP found: {len(res_siglip)} results")
for i, point in enumerate(res_siglip[:3]):
    print(f"  {i+1}. Score: {point.score:.4f}, Video: {point.payload.get('video_id')}, Frame: {point.payload.get('frame_idx')}")
