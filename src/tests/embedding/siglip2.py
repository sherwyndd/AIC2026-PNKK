import torch
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-so400m-patch14-384"


def load_siglip2(device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Load model
    print("Loading model...")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=False,
    )

    # Move model to device
    model = model.to(device)

    # Evaluation mode
    model.eval()

    print("SigLIP2 loaded successfully.")

    return processor, model, device