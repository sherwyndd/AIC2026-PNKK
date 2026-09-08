from PIL import Image
import torch

import os
import sys

try:
    from .siglip2 import load_siglip2
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from embedding.siglip2 import load_siglip2

processor, model, device = load_siglip2()

image = Image.open(
    "/AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/keyframes/K01_V001/K01_V001_s0001_k01.jpg"
).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt",
)

inputs = {
    k: v.to(device, non_blocking=True)
    for k, v in inputs.items()
}

with torch.inference_mode():
    outputs = model.get_image_features(**inputs)

# Lấy embedding
if isinstance(outputs, torch.Tensor):
    features = outputs
elif hasattr(outputs, "pooler_output"):
    features = outputs.pooler_output
else:
    raise TypeError(f"Unexpected output type: {type(outputs)}")

# Normalize
features = torch.nn.functional.normalize(features, dim=-1)

print("Model :", type(model))
print("Shape :", features.shape)
print("Device:", features.device)
print("Dtype :", features.dtype)
print("Norm  :", features.norm(dim=-1))
print(model.config._name_or_path)
print(model.config.model_type)