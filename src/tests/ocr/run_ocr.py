from __future__ import annotations

import gc
import random
import json
import os
import sys
import time
from pathlib import Path
import cv2
import paddle
import yaml
from paddleocr import PaddleOCR
from tqdm import tqdm

# Ensure we can run from anywhere inside the workspace
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.chdir(project_root)

from src.utils.dataset_utils import (
    KEYFRAME_ROOT,
    METADATA_KEYFRAME_ROOT,
    get_completed_videos,
    is_metadata_keyframes_done,
)

INITIAL_BATCH_SIZE = 16  # High starting batch size for PaddleOCR speed

LOCK_DIR = Path("outputs/ocr/.locks")
LOCK_DIR.mkdir(parents=True, exist_ok=True)

def is_ocr_done(video_name: str, ocr_output_dir: Path) -> bool:
    """Check if a video has already been processed by OCR."""
    out_file = ocr_output_dir / f"{video_name}.json"
    return out_file.exists() and out_file.stat().st_size > 0

def get_lock_path(video_name: str) -> Path:
    return LOCK_DIR / f"{video_name}.lock"


def acquire_lock(video_name: str) -> bool:
    lock_path = get_lock_path(video_name)

    try:
        # atomic create
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(video_name: str):
    lock_path = get_lock_path(video_name)
    if lock_path.exists():
        lock_path.unlink()



def clear_gpu_memory():
    """Clear Paddle and Python GPU memory cache."""
    gc.collect()
    try:
        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def parse_page_result(res, conf_threshold: float, scale_factor: float = 1.0):
    """Extract texts, scores, and polygons from a PaddleOCR prediction result."""
    texts = []
    merged_text = []

    if not res:
        return texts, ""

    page = res[0] if (isinstance(res, list) and len(res) > 0) else (res if isinstance(res, dict) else {})
    if not isinstance(page, dict):
        return texts, ""

    rec_texts = page.get("rec_texts", [])
    rec_scores = page.get("rec_scores", [])
    dt_polys = page.get("dt_polys", [])

    for poly, text, score in zip(dt_polys, rec_texts, rec_scores):
        text = text.strip()
        score = float(score)

        if not text or score < conf_threshold:
            continue

        poly = poly.tolist() if hasattr(poly, "tolist") else poly

        if scale_factor != 1.0 and poly:
            inv_scale = 1.0 / scale_factor
            poly = [[pt[0] * inv_scale, pt[1] * inv_scale] for pt in poly]

        texts.append(
            {
                "text": text,
                "score": score,
                "polygon": poly,
            }
        )
        merged_text.append(text)

    return texts, " ".join(merged_text)


def process_video_ocr(keyframe_dir: Path, ocr_output_dir: Path, ocr: PaddleOCR, conf_threshold: float):
    video_name = keyframe_dir.name
    current_batch_size = INITIAL_BATCH_SIZE  # Reset batch_size for each video

    print(f"\n========== OCR: {video_name} ==========")
    print(f"  [0/4] Starting OCR pipeline for: {video_name}")

    print(f"  [1/4] Scanning images in: {keyframe_dir}")
    image_paths = sorted(keyframe_dir.rglob("*.jpg"))
    print(f"  [1/4] Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        print(f"  [SKIP] Empty folder, skipping {video_name}")
        return

    print(f"  [2/4] Running OCR with initial batch_size={current_batch_size}...")

    results = []
    i = 0
    pbar = tqdm(total=len(image_paths), desc=video_name)

    while i < len(image_paths):
        batch_paths = image_paths[i : i + current_batch_size]

        images = []
        metadata = []  # stores (rel_path, h, w)

        for path in batch_paths:
            rel_path = str(path)
            img = cv2.imread(rel_path)
            if img is None:
                print(f"[ERROR] Cannot read image: {rel_path}")
                continue
            h, w = img.shape[:2]
            images.append(img)
            metadata.append((rel_path, h, w))

        if not images:
            i += len(batch_paths)
            pbar.update(len(batch_paths))
            continue

        try:
            # Predict batch or single image
            if len(images) == 1:
                batch_res = [ocr.predict(images[0])]
            else:
                batch_res = ocr.predict(images)

            if not isinstance(batch_res, list):
                batch_res = [batch_res]

            for k, (rel_path, h, w) in enumerate(metadata):
                res = batch_res[k] if k < len(batch_res) else None
                texts, merged_str = parse_page_result(res, conf_threshold)

                results.append(
                    {
                        "doc_id": len(results),
                        "image": rel_path,
                        "width": w,
                        "height": h,
                        "texts": texts,
                        "text": merged_str,
                    }
                )

            batch_len = len(batch_paths)
            pbar.update(batch_len)
            i += batch_len

        except Exception as e:
            err_msg = str(e).lower()
            is_oom = "out of memory" in err_msg or "cuda" in err_msg or "memory" in err_msg

            if is_oom:
                clear_gpu_memory()

                if current_batch_size > 1:
                    new_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"\n[OOM WARNING] PaddleOCR CUDA OOM with batch_size={current_batch_size}! "
                        f"Auto-reducing batch_size to {new_batch_size} and retrying..."
                    )
                    current_batch_size = new_batch_size
                    # Retrying current batch with reduced batch size
                else:
                    # Fallback single image 50% scale
                    rel_path, h, w = metadata[0]
                    print(f"\n[OOM WARNING] Single image OOM on {rel_path}, downscaling 50%...")
                    scale_factor = 0.5
                    new_w, new_h = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
                    resized_img = cv2.resize(images[0], (new_w, new_h), interpolation=cv2.INTER_AREA)
                    try:
                        res = ocr.predict(resized_img)
                        texts, merged_str = parse_page_result(res, conf_threshold, scale_factor=scale_factor)
                    except Exception as e2:
                        print(f"[OOM ERROR] Failed to process {rel_path}: {e2}")
                        texts, merged_str = [], ""

                    results.append(
                        {
                            "doc_id": len(results),
                            "image": rel_path,
                            "width": w,
                            "height": h,
                            "texts": texts,
                            "text": merged_str,
                        }
                    )
                    i += 1
                    pbar.update(1)
            else:
                print(f"\n[ERROR] Batch processing error: {e}")
                i += len(batch_paths)
                pbar.update(len(batch_paths))

    pbar.close()
    print(f"  [2/4] OCR inference done. {len(results)} results collected.")

    print(f"  [3/4] Writing JSON output...")
    output_file = ocr_output_dir / f"{video_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  [3/4] Saved {len(results)} images -> {output_file}")
    print(f"[DONE] {video_name}")

    # Explicit memory cleanup after video
    print(f"  [cleanup] Freeing memory after {video_name}...")
    del results
    clear_gpu_memory()
    print(f"  [cleanup] Memory freed.")


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path.resolve()}")
        sys.exit(1)

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # OCR config
    ocr_config = config.get("ocr", {})
    lang = ocr_config.get("language", "vi")
    if lang == "vie":
        lang = "vi"
    conf_threshold = ocr_config.get("confidence_threshold", 0.5)

    # Paths
    paths_config = config.get("paths", {})
    keyframe_root = Path(paths_config.get("keyframe_root", KEYFRAME_ROOT))
    output_dir = Path(paths_config.get("output_dir", "outputs"))
    ocr_output_dir = output_dir / "ocr"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    check_interval = config.get("pipeline", {}).get("check_interval", 30)

    print("[INIT] run_ocr.py starting up...")
    print(f"[INIT] Project root: {project_root}")
    print(f"[INIT] Keyframe root: {keyframe_root}")
    print(f"[INIT] Metadata keyframe root: {METADATA_KEYFRAME_ROOT}")
    print(f"[INIT] OCR output dir: {ocr_output_dir}")
    print(f"[INIT] Language: {lang} | Conf threshold: {conf_threshold}")
    print(f"[INIT] Initial batch size: {INITIAL_BATCH_SIZE}")
    print(f"[INIT] Check interval: {check_interval}s")

    # PaddleOCR
    use_gpu = paddle.is_compiled_with_cuda()
    print(f"[INIT] Initializing PaddleOCR ({'GPU' if use_gpu else 'CPU'})...")
    print(f"[INIT] Device: {paddle.device.get_device()}")
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=1536,
    )
    print(f"[INIT] PaddleOCR model loaded.")

    print(
        f"[INIT] Starting continuous OCR processing loop with adaptive batching & memory cleanup "
        f"(Initial batch_size: {INITIAL_BATCH_SIZE}, checking '{keyframe_root}' with metadata in '{METADATA_KEYFRAME_ROOT}')..."
    )

    try:
        while True:
            print(f"\n[SCAN] Scanning keyframe root: {keyframe_root} (checking metadata in {METADATA_KEYFRAME_ROOT})")
            video_names = get_completed_videos(keyframe_root, check_metadata=True)
            print(f"[SCAN] Total keyframe folders with completed metadata found: {len(video_names)}")


            already_done = [v for v in video_names if is_ocr_done(v, ocr_output_dir)]
            pending_videos = [v for v in video_names if not is_ocr_done(v, ocr_output_dir)]
            random.shuffle(pending_videos)
            print(f"[SCAN] Already OCR'd: {len(already_done)} | Pending: {len(pending_videos)}")

            if pending_videos:
                print(f"[LOOP] Pending videos: {pending_videos}")
                for idx, video_name in enumerate(pending_videos, 1):

                    if not acquire_lock(video_name):
                        print(f"[LOCK] {video_name} is being processed by another worker.")
                        continue

                    if is_ocr_done(video_name, ocr_output_dir):
                        print(f"[SKIP] {video_name} was OCR'd by another worker.")
                        release_lock(video_name)
                        continue

                    try:
                        print(f"\n[LOOP] Processing {idx}/{len(pending_videos)}: {video_name}")

                        keyframe_dir = keyframe_root / video_name

                        process_video_ocr(
                            keyframe_dir,
                            ocr_output_dir,
                            ocr,
                            conf_threshold,
                        )

                    finally:
                        release_lock(video_name)
            else:
                print(
                    f"[WAIT] No new keyframe folders to OCR. Sleeping {check_interval}s..."
                )
                time.sleep(check_interval)
    except KeyboardInterrupt:
        print("\nOCR service stopped by user.")


if __name__ == "__main__":
    main()
