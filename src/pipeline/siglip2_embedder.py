import gc
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-so400m-patch14-384"


class KeyframeImageDataset(Dataset):
    """Dataset for loading keyframe images for SigLIP2 feature extraction."""

    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[str, Any]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                return str(path), img
        except Exception:
            # Fallback to black image if corrupt
            return str(path), Image.new("RGB", (384, 384), (0, 0, 0))


def collate_fn_pil(batch: List[Tuple[str, Image.Image]]) -> Tuple[List[str], List[Image.Image]]:
    paths, images = zip(*batch)
    return list(paths), list(images)


class SigLIP2Embedder:
    """SigLIP2 image feature extractor."""

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = None
        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.processor, self.model

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=False,
        ).to(self.device)
        self.model.eval()
        return self.processor, self.model

    def clear_memory(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def embed_keyframe_dir(
        self,
        keyframe_dir: Path,
        output_npy: Path,
        batch_size: int = 128,
        num_workers: int = 4,
        logger: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """Extract SigLIP2 embeddings for all keyframe images in keyframe_dir and save .npy."""
        processor, model = self.load_model()
        image_paths = sorted(Path(keyframe_dir).rglob("*.jpg"))
        if not image_paths:
            image_paths = sorted(Path(keyframe_dir).rglob("*.png"))

        if not image_paths:
            if logger:
                logger.info(f"No keyframe images found in {keyframe_dir}")
            return None

        dataset = KeyframeImageDataset(image_paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if os.name != "nt" else 0,
            collate_fn=collate_fn_pil,
            pin_memory=(self.device.type == "cuda"),
        )

        all_embeddings = []
        pbar = tqdm(total=len(image_paths), desc=f"Embedding {keyframe_dir.name}")

        with torch.no_grad():
            for paths, images in loader:
                try:
                    inputs = processor(images=images, return_tensors="pt").to(self.device)
                    if self.device.type == "cuda":
                        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

                    image_features = model.get_image_features(**inputs)
                    # Normalize embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    embeddings = image_features.cpu().numpy().astype(np.float32)
                    all_embeddings.append(embeddings)
                    pbar.update(len(paths))
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if logger:
                            logger.warning("OOM detected in SigLIP2 batching, reducing batch size...")
                        self.clear_memory()
                        # Fallback element by element for this batch
                        for img in images:
                            inp = processor(images=[img], return_tensors="pt").to(self.device)
                            if self.device.type == "cuda":
                                inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
                            feat = model.get_image_features(**inp)
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                            all_embeddings.append(feat.cpu().numpy().astype(np.float32))
                            pbar.update(1)
                    else:
                        raise exc

        pbar.close()

        if not all_embeddings:
            return None

        final_matrix = np.concatenate(all_embeddings, axis=0)
        output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_npy, final_matrix)
        self.clear_memory()
        return final_matrix
