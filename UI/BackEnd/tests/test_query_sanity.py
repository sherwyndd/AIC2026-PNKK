"""Quick accuracy test with a single query."""

import sys
sys.path.insert(0, "/AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
from qdrant_client import QdrantClient
from models.siglip_encoder import load_siglip, encode_text_siglip

COLLECTION_NAME = "image_siglip"
QUERY = "two people talking on the street"
TOP_K = 10

print("Loading SigLIP2...")
load_siglip(device="cuda:0")

client = QdrantClient(host="localhost", port=6333)
info = client.get_collection(COLLECTION_NAME)
print(f"Collection: {info.points_count} vectors (dim={info.config.params.vectors.size})")

print(f"\nQuery: \"{QUERY}\"")
vec = encode_text_siglip(QUERY)
print(f"Vector dim: {vec.shape[0]} | L2 norm: {np.linalg.norm(vec):.4f}")

result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=vec.tolist(),
    limit=TOP_K,
)
hits = result.points

if not hits:
    print("No results.")
else:
    scores = [h.score for h in hits]
    print(f"\nTop {TOP_K} results (score range: {min(scores):.4f} ~ {max(scores):.4f}):")
    for i, h in enumerate(hits):
        p = h.payload
        print(f"  {i+1:2d}. [{h.score:.4f}] {p.get('video_id')} frame={p.get('frame_idx'):5d} → {p.get('image_path')}")
