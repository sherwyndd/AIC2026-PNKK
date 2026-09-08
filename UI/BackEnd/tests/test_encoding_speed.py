"""Test encoding speed to estimate re-index time."""

import time
from pathlib import Path
from PIL import Image
from models.siglip_encoder import load_siglip

KEYFRAME_ROOT = "/AIClub_NAS/core_baotg/nhan/dataset/keyframes"
DEVICE = "cuda:0"

def encode_image_siglip(image_path: str):
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
    return image_emb.astype("float32")

# Load model
print("Loading SigLIP model...")
load_siglip(device=DEVICE)

# Get sample images
video_dir = Path(KEYFRAME_ROOT) / "L24_V005"
if not video_dir.exists():
    video_dir = list(Path(KEYFRAME_ROOT).iterdir())[0]

sample_images = list(video_dir.glob("*.jpg"))[:100]
print(f"Testing with {len(sample_images)} images...")

# Measure encoding time
start = time.time()
for img_path in sample_images:
    encode_image_siglip(str(img_path))
elapsed = time.time() - start

avg_time = elapsed / len(sample_images)
print(f"\n=== Encoding Speed ===")
print(f"Total time for {len(sample_images)} images: {elapsed:.2f}s")
print(f"Average time per image: {avg_time*1000:.2f}ms")

# Estimate total time
total_images = 274542
remaining = total_images - 5888  # already indexed
total_estimated = remaining * avg_time

print(f"\n=== Time Estimate ===")
print(f"Total images: {total_images}")
print(f"Already indexed: 5,888")
print(f"Remaining: {remaining}")
print(f"Estimated time for remaining: {total_estimated/60:.2f} minutes ({total_estimated/3600:.2f} hours)")
