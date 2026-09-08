"""Quick sanity check: test text-to-image search on current collection.

Loads the FIXED siglip_encoder (correct extraction order) and
runs a few queries to verify cross-modal alignment.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
import sys
sys.path.insert(0, "/AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd")

import numpy as np
from qdrant_client import QdrantClient

from models.siglip_encoder import load_siglip, encode_text_siglip

COLLECTION = "image_siglip"
TOP_K = 5

# 1. Load model
print("=" * 70)
print("Loading SigLIP2 model ...")
load_siglip(device="cuda:0")
print("Model loaded.\n")

# 2. Connect to Qdrant and check state
client = QdrantClient(host="127.0.0.1", port=6333, timeout=60)
info = client.get_collection(COLLECTION)
print(f"Collection state:")
print(f"  points_count = {info.points_count}")
print(f"  dim          = {info.config.params.vectors.size}")
print(f"  distance     = {info.config.params.vectors.distance}")
print(f"  (30 folders first: expect ~30*500 ≈ 15,000 points)\n")

# 3. Sanity queries (simple common concepts)
QUERIES = [
    "a tall building",
    "person walking on street",
    "car on the road",
    "tree and green plants",
    "a motorcycle",
    "people talking",
]

print("=" * 70)
for q in QUERIES:
    vec = encode_text_siglip(q)
    norm = float(np.linalg.norm(vec))
    res = client.query_points(collection_name=COLLECTION, query=vec.tolist(), limit=TOP_K)
    hits = res.points if hasattr(res, "points") else res

    scores = [float(h.score) for h in hits]
    top1_vid = hits[0].payload.get("video_id") if hits else "N/A"
    top1_fr  = hits[0].payload.get("frame_idx") if hits else "N/A"
    top1_p   = hits[0].payload.get("image_path") if hits else "N/A"

    print(f"Q: {q!r:35s} | L2={norm:.4f}")
    print(f"   Top score: {max(scores):.4f}  (Video {top1_vid} frame {top1_fr})")
    print(f"   Score range: {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"   Top1 image: {top1_p}")
    # Interpretation
    if max(scores) < 0.08:
        print(f"   ⚠️  Score < 0.08 → RANDOM alignment still! (PROBLEM)")
    elif max(scores) < 0.15:
        print(f"   🟡 Score 0.08-0.15 → Weak alignment")
    elif max(scores) < 0.25:
        print(f"   🟢 Score 0.15-0.25 → OK-ish alignment")
    else:
        print(f"   ✅ Score > 0.25 → GOOD cross-modal alignment!")
    print("-" * 70)

print("\nSanity check DONE!")
