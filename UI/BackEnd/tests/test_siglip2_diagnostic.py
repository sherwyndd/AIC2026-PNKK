"""Deep diagnostic: verify SigLIP2 text/image embedding alignment."""

import sys
sys.path.insert(0, "/AIClub_NAS/core_baotg/phong/AIC_2026/UI/BackEnd")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient

print("=" * 60)
print("STEP 1: Load SigLIP2 raw model (AutoModel)")
from transformers import AutoModel, AutoProcessor

DEVICE = "cuda:0"
MODEL_ID = "google/siglip2-base-patch16-224"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print(f"  Model type: {type(model).__name__}")
print(f"  Has get_text_features: {hasattr(model, 'get_text_features')}")
print(f"  Has get_image_features: {hasattr(model, 'get_image_features')}")

# ── STEP 2: Check text feature extraction ────────────────────────────────────
print("\nSTEP 2: Encode text and check output type")
text = "a person riding a motorcycle on a busy street"
t_inputs = processor(text=[text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    t_out = model.get_text_features(**t_inputs)

print(f"  Text output type: {type(t_out)}")
if isinstance(t_out, torch.Tensor):
    print(f"  Direct tensor! Shape: {t_out.shape}")
    t_feat = t_out
else:
    print(f"  ModelOutput! Attrs: {[k for k in dir(t_out) if not k.startswith('_')]}")
    if hasattr(t_out, 'pooler_output') and t_out.pooler_output is not None:
        t_feat = t_out.pooler_output
        print(f"  Using pooler_output, shape: {t_feat.shape}")
    elif hasattr(t_out, 'text_embeds') and t_out.text_embeds is not None:
        t_feat = t_out.text_embeds
        print(f"  Using text_embeds, shape: {t_feat.shape}")
    else:
        t_feat = t_out[0]
        print(f"  Using [0], shape: {t_feat.shape}")

t_feat_norm = F.normalize(t_feat, p=2, dim=-1)[0].cpu().float().numpy()
print(f"  Final text embedding L2 norm: {np.linalg.norm(t_feat_norm):.4f}")

# ── STEP 3: Fetch one stored vector from Qdrant and compute similarity ────────
print("\nSTEP 3: Fetch vector from Qdrant and compute cosine similarity with text")
client = QdrantClient(host="localhost", port=6333)
# Get a sample point with its vector
sample = client.query_points(
    collection_name="image_siglip",
    query=t_feat_norm.tolist(),
    limit=5,
    with_vectors=True,
)
hits = sample.points
print(f"  Got {len(hits)} hits. Top score from Qdrant: {hits[0].score if hits else 'N/A'}")

if hits and hits[0].vector:
    stored_vec = np.array(hits[0].vector, dtype=np.float32)
    manual_dot = float(np.dot(t_feat_norm, stored_vec))
    manual_l2 = np.linalg.norm(stored_vec)
    print(f"  Stored vector L2 norm: {manual_l2:.4f}")
    print(f"  Manual cosine dot product: {manual_dot:.4f}  (should match Qdrant score)")

# ── STEP 4: Directly encode an image from keyframes and compare ───────────────
print("\nSTEP 4: Encode a real keyframe image and compare with matching text")
img_path = f"/AIClub_NAS/core_baotg/nhan/dataset/keyframes/{hits[0].payload.get('image_path')}"
print(f"  Image: {img_path}")

try:
    img = Image.open(img_path).convert("RGB")
    i_inputs = processor(images=[img], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        i_out = model.get_image_features(**i_inputs)

    print(f"  Image output type: {type(i_out)}")
    if isinstance(i_out, torch.Tensor):
        i_feat = i_out
        print(f"  Direct tensor! Shape: {i_feat.shape}")
    else:
        print(f"  ModelOutput! Attrs: {[k for k in dir(i_out) if not k.startswith('_')]}")
        if hasattr(i_out, 'pooler_output') and i_out.pooler_output is not None:
            i_feat = i_out.pooler_output
            print(f"  Using pooler_output, shape: {i_feat.shape}")
        elif hasattr(i_out, 'image_embeds') and i_out.image_embeds is not None:
            i_feat = i_out.image_embeds
            print(f"  Using image_embeds, shape: {i_feat.shape}")
        else:
            i_feat = i_out[0]
            print(f"  Using [0], shape: {i_feat.shape}")

    i_feat_norm = F.normalize(i_feat, p=2, dim=-1)[0].cpu().float().numpy()
    print(f"  Image embedding L2 norm: {np.linalg.norm(i_feat_norm):.4f}")

    # Cross cosine similarity
    cross_sim = float(np.dot(t_feat_norm, i_feat_norm))
    print(f"\n  ✅ Text vs Image (freshly encoded) cosine similarity: {cross_sim:.4f}")
    if stored_vec is not None:
        stored_sim = float(np.dot(i_feat_norm, stored_vec))
        print(f"  ✅ Freshly encoded image vs stored vector similarity: {stored_sim:.4f}")
        if stored_sim > 0.999:
            print("  ✅ Stored vectors MATCH freshly encoded! No mismatch.")
        else:
            print("  ⚠️  Stored vectors DO NOT match freshly encoded! MISMATCH DETECTED.")

    if cross_sim > 0.15:
        print(f"\n  ✅ Cross-modal alignment OK: score {cross_sim:.4f} > 0.15")
    elif cross_sim > 0.05:
        print(f"\n  ⚠️  Alignment is weak (score={cross_sim:.4f}). Collection may still be building or query is poor match.")
    else:
        print(f"\n  ❌ Alignment FAILED: score {cross_sim:.4f} is near-random. Embedding space mismatch!")

except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
print("Diagnosis DONE!")
