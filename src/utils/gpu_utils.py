import torch


def free_vram():
    """Release cached GPU memory after a pipeline stage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
