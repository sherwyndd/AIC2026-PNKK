from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

try:
    from .siglip2 import load_siglip2
except ImportError:
    try:
        from src.embedding.siglip2 import load_siglip2
    except ImportError:
        from embedding.siglip2 import load_siglip2

from src.utils.dataset_utils import KEYFRAME_ROOT, METADATA_KEYFRAME_ROOT, get_completed_videos, is_metadata_keyframes_done

OUTPUT_ROOT = Path("outputs/siglip2_final")
INITIAL_BATCH_SIZE = 256  # Start high for maximum performance
NUM_WORKERS = 4  # Number of background worker processes for loading & preprocessing
PREFETCH_FACTOR = 2  # Prefetch factor per worker (when NUM_WORKERS > 0)
CHECK_INTERVAL = 30  # seconds


@contextmanager
def video_file_lock(video_name: str, output_root: Path):
    """File lock using atomic creation (O_CREAT | O_EXCL).

    Skips if another process holds the lock and always cleans up on exit/exception.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / f"{video_name}.lock"

    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"pid={os.getpid()} time={time.time()}\n")
        acquired = True
    except (FileExistsError, OSError):
        acquired = False

    try:
        yield acquired
    finally:
        if acquired:
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except OSError:
                pass


def is_embed_done(video_name: str, output_root: Path) -> bool:
    """Check if a video's image embeddings have already been generated and are valid."""
    video_out = output_root / video_name
    npy_file = video_out / "image_embeddings.npy"
    txt_file = video_out / "image_paths.txt"
    if not (
        npy_file.exists()
        and npy_file.stat().st_size > 0
        and txt_file.exists()
        and txt_file.stat().st_size > 0
    ):
        return False

    try:
        # Quick check to ensure file is not corrupted
        loaded = np.load(npy_file, mmap_mode="r")
        return bool(loaded.shape[0] > 0)
    except Exception:
        return False


def load_images(keyframe_dir: Path) -> list[Path]:
    """Find all JPG images recursively."""
    print(f"  [1/5] Scanning images in: {keyframe_dir}")
    image_paths = sorted(keyframe_dir.rglob("*.jpg"))
    print(f"  [1/5] Found {len(image_paths)} .jpg images.")
    return image_paths


class KeyframeDataset(Dataset):
    """Dataset that reads images in parallel and runs processor preprocessing in worker processes."""

    def __init__(self, image_paths: list[Path], processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
            return pixel_values, p, True
        except Exception as e:
            return None, p, False


def collate_fn(batch):
    """Collate function that filters corrupted images and stacks pixel tensors."""
    valid_tensors = []
    valid_paths = []
    for tensor, path, ok in batch:
        if ok and tensor is not None:
            valid_tensors.append(tensor)
            valid_paths.append(path)
        else:
            print(f"    [WARN] Skip unreadable image: {path}")

    if not valid_tensors:
        return None, []

    batch_tensors = torch.stack(valid_tensors, dim=0)
    return batch_tensors, valid_paths


def create_dataloader(
    image_paths: list[Path],
    processor,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """Create optimized DataLoader with persistent workers.

    Note: pin_memory is disabled because images are on NAS (NFS mount),
    which causes 'Pin memory thread exited unexpectedly' with multiprocessing workers.
    """
    dataset = KeyframeDataset(image_paths, processor)
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Must be False on NFS/NAS storage to avoid pin_memory thread crash
        persistent_workers=use_workers,
        prefetch_factor=PREFETCH_FACTOR if use_workers else None,
        collate_fn=collate_fn,
    )


def encode_batch(batch_tensors, model, device):
    """Encode one batch of image tensors and return NumPy array."""
    target_dtype = (
        model.dtype
        if hasattr(model, "dtype")
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    pixel_values = batch_tensors.to(device, dtype=target_dtype, non_blocking=True)

    with torch.inference_mode():
        outputs = model.get_image_features(pixel_values=pixel_values)

        if isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            features = outputs.pooler_output

        features = torch.nn.functional.normalize(
            features,
            dim=-1,
        )

    # Convert to numpy on CPU immediately so PyTorch graph/tensors are released
    return features.cpu().numpy()


def save_embeddings(embeddings, image_paths, output_dir):
    """Save embeddings and corresponding image paths atomically and verify."""
    print(f"  [5/5] Saving embeddings to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_final = output_dir / "image_embeddings.npy"
    txt_final = output_dir / "image_paths.txt"
    npy_tmp = output_dir / "image_embeddings_tmp.npy"
    txt_tmp = output_dir / "image_paths_tmp.txt"

    try:
        # 1. Write to temporary files first
        with open(npy_tmp, "wb") as f:
            np.save(f, embeddings)

        with open(txt_tmp, "w", encoding="utf-8") as f:
            for p in image_paths:
                f.write(str(p) + "\n")

        # 2. Atomic replacement
        os.replace(npy_tmp, npy_final)
        os.replace(txt_tmp, txt_final)

        # 3. Post-save verification
        loaded = np.load(npy_final, mmap_mode="r")
        if loaded.shape != embeddings.shape:
            raise ValueError(
                f"Embedding shape mismatch: expected {embeddings.shape}, got {loaded.shape}"
            )

        with open(txt_final, "r", encoding="utf-8") as f:
            saved_lines = sum(1 for line in f if line.strip())
        if saved_lines != len(image_paths):
            raise ValueError(
                f"Path count mismatch: expected {len(image_paths)}, got {saved_lines}"
            )

        print(f"  [5/5] Embedding shape: {embeddings.shape} (verified)")
        print(f"  [5/5] Saved to: {output_dir}")

    except Exception as e:
        print(f"  [ERROR] Failed to save/verify embeddings in {output_dir}: {e}")
        # Clean up temporary and corrupted files
        for f_path in (npy_tmp, txt_tmp, npy_final, txt_final):
            try:
                if f_path.exists():
                    f_path.unlink()
            except OSError:
                pass
        raise e


def process_video_embed(keyframe_dir: Path, processor, model, device, output_root: Path):
    video_name = keyframe_dir.name

    with video_file_lock(video_name, output_root) as acquired:
        if not acquired:
            print(f"  [SKIP] Video {video_name} is locked by another process.")
            return

        current_batch_size = INITIAL_BATCH_SIZE  # Reset batch size for each video

        print(f"\n========== EMBEDDING: {video_name} ==========")
        print(f"  [0/5] Starting pipeline for: {video_name}")

        image_paths = load_images(keyframe_dir)
        if len(image_paths) == 0:
            print(f"  [SKIP] Empty folder, skipping {video_name}")
            return

        print(
            f"  [2/5] Encoding {len(image_paths)} images "
            f"(batch_size={current_batch_size}, num_workers={NUM_WORKERS}, device={device})..."
        )

        all_features = []
        all_valid_paths = []

        processed_idx = 0
        pbar = tqdm(total=len(image_paths), desc=video_name)

        while processed_idx < len(image_paths):
            remaining_paths = image_paths[processed_idx:]
            loader = create_dataloader(
                remaining_paths,
                processor,
                batch_size=current_batch_size,
                num_workers=NUM_WORKERS,
            )

            try:
                for batch_tensors, valid_paths in loader:
                    if batch_tensors is not None and len(valid_paths) > 0:
                        features = encode_batch(batch_tensors, model, device)
                        all_features.append(features)
                        all_valid_paths.extend(valid_paths)
                        del features

                    batch_count = len(valid_paths) if valid_paths else (len(batch_tensors) if batch_tensors is not None else 1)
                    processed_idx += batch_count
                    pbar.update(batch_count)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                err_msg = str(e).lower()
                is_oom = (
                    "out of memory" in err_msg
                    or "cuda error" in err_msg
                    or (hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError))
                )

                if is_oom:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if current_batch_size > 1:
                        new_batch_size = max(1, current_batch_size // 2)
                        print(
                            f"\n[OOM WARNING] CUDA Out of Memory with batch_size={current_batch_size}! "
                            f"Auto-reducing batch_size to {new_batch_size} and retrying remaining..."
                        )
                        current_batch_size = new_batch_size
                        # Continues loop and creates loader for remaining images with reduced batch size
                    else:
                        print(f"\n[OOM ERROR] OOM at batch_size=1! Skipping image: {image_paths[processed_idx]}")
                        processed_idx += 1
                        pbar.update(1)
                else:
                    print(f"\n[ERROR] Batch processing error on video {video_name}: {e}")
                    pbar.close()
                    raise e

        pbar.close()
        print(f"  [3/5] Encoding done. Collected {len(all_features)} batches ({len(all_valid_paths)} valid images).")

        if not all_features:
            print(f"  [SKIP] No valid images in {video_name}")
            return

        print(f"  [4/5] Concatenating features into final embedding array...")
        embeddings = np.concatenate(all_features, axis=0)
        output_dir = output_root / video_name
        save_embeddings(embeddings, all_valid_paths, output_dir)
        print(f"[DONE] {video_name}")

        # Explicit memory cleanup after video processing
        print(f"  [cleanup] Freeing memory after {video_name}...")
        del embeddings
        del all_features
        del all_valid_paths
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [cleanup] Memory freed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Continuous Keyframe Embedding Service using SigLIP 2")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g. 'cuda:0', 'cuda:1', 'cpu'). Default is first available CUDA device.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of background DataLoader worker processes (default: {NUM_WORKERS})",
    )
    parser.add_argument(
        "--keyframe_root",
        type=Path,
        default=KEYFRAME_ROOT,
        help=f"Directory containing keyframe folders (default: {KEYFRAME_ROOT})",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Directory to save embeddings (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process available pending videos once and exit.",
    )
    return parser.parse_args()


def main():
    global NUM_WORKERS

    args = parse_args()
    NUM_WORKERS = args.num_workers
    keyframe_root = args.keyframe_root
    output_root = args.output_root

    # Determine target device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[INIT] embed_images.py starting up...")
    print(f"[INIT] Project root: {project_root}")
    print(f"[INIT] Keyframe root: {keyframe_root}")
    print(f"[INIT] Output root: {output_root}")
    print(f"[INIT] Target device: {device}")
    print(f"[INIT] Initial batch size: {INITIAL_BATCH_SIZE}")
    print(f"[INIT] Num workers: {NUM_WORKERS} | Prefetch factor: {PREFETCH_FACTOR}")
    print(f"[INIT] Check interval: {CHECK_INTERVAL}s")
    print(f"[INIT] Loading SigLIP 2 model on {device}...")
    processor, model, device = load_siglip2(device=device)
    print(f"[INIT] Model loaded on device: {device}")

    print(
        f"[INIT] Starting continuous embedding loop on {device} with adaptive batching & memory cleanup "
        f"(Initial batch_size: {INITIAL_BATCH_SIZE}, checking '{keyframe_root}')..."
    )

    try:
        while True:
            print(f"\n[SCAN] Scanning keyframe root: {keyframe_root}")
            if keyframe_root == KEYFRAME_ROOT:
                video_names = get_completed_videos(keyframe_root, check_metadata=True)
            else:
                video_names = [d.name for d in sorted(keyframe_root.iterdir()) if d.is_dir()] if keyframe_root.exists() else []

            print(f"[SCAN] Total keyframe folders found: {len(video_names)}")

            already_done = [v for v in video_names if is_embed_done(v, output_root)]
            pending_videos = [v for v in video_names if not is_embed_done(v, output_root)]
            print(f"[SCAN] Already embedded: {len(already_done)} | Pending: {len(pending_videos)}")

            if pending_videos:
                print(f"[LOOP] Pending videos: {pending_videos}")
                for idx, video_name in enumerate(pending_videos, 1):
                    print(f"\n[LOOP] [{device}] Processing {idx}/{len(pending_videos)}: {video_name}")
                    keyframe_dir = keyframe_root / video_name
                    process_video_embed(keyframe_dir, processor, model, device, output_root)

            if args.once:
                print("[ONCE] Finished single scan pass. Exiting.")
                break

            if not pending_videos:
                print(
                    f"[WAIT] No new keyframe folders to embed. Sleeping {CHECK_INTERVAL}s..."
                )
                time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nEmbedding service stopped by user.")


if __name__ == "__main__":
    main()