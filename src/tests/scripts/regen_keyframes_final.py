#!/usr/bin/env python3
"""
regen_keyframes_final.py
========================
Trích xuất keyframe lại từ shot metadata có sẵn, dùng:
  - OpenCLIP ViT-L-14-quickgelu pretrained DFN2B (đúng paper Vortex)
  - L2-normalized rel_diff filtering (fix bug code cũ)
  - Multi-GPU: mỗi GPU chạy trong process riêng, xử lý tuần tự video theo lượt

INPUT:
  --video_dirs  : thư mục chứa *.mp4, ngăn bởi ':', script sẽ scan recursive
                  vd: /dir1:/dir2:/dir3
  --shots_dir   : thư mục chứa *_shots.csv
  --videos_meta : file videos_metadata.csv (để lấy fps)
  --output_dir  : thư mục đầu ra (images/ + metadata/)
  --gpus        : danh sách GPU IDs, vd: 1,2,3,5,6
  --batch_size  : frames/batch mỗi lần encode CLIP (default 16)
  --threshold   : rel_diff threshold (default 0.4, theo paper Vortex)
  --sample_step : cứ N frame lấy 1 candidate (default 8)
  --force       : chạy lại kể cả đã có output

CÁCH CHẠY CƠ BẢN (chỉ chạy video loại L):
  python scripts/regen_keyframes_final.py \
      --video_dirs /mlcv1/Datasets/HCMAI25/full \
      --shots_dir /AIClub_NAS/core_baotg/nhan/dataset/metadata/shots \
      --videos_meta /AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv \
      --output_dir /AIClub_NAS/core_baotg/nhan/dataset/keyframes_for_final \
      --gpus 1,2,3,5,6

RESUME:
  Chạy lại cùng lệnh — video đã có metadata CSV sẽ bị skip tự động.
  Dùng --force để chạy lại từ đầu.
"""

from __future__ import annotations

import os
import sys
import warnings

# ── SUPPRESS HARMLESS WARNINGS (FFmpeg / OpenCV / HuggingFace) ──
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["FFMPEG_LOGLEVEL"] = "quiet"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import argparse
import csv
import glob
import logging
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("open_clip").setLevel(logging.ERROR)
try:
    import huggingface_hub.utils._http
    huggingface_hub.utils._http.logger.setLevel(logging.ERROR)
except Exception:
    pass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class SuppressCStderr:
    """Redirect C-level stderr (fd 2) to devnull to mute FFmpeg C-level decoder warnings."""
    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_WRONLY)
            self.save_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.null_fd = None
            self.save_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_fd is not None and self.null_fd is not None:
            try:
                os.dup2(self.save_fd, 2)
                os.close(self.save_fd)
                os.close(self.null_fd)
            except Exception:
                pass

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
OPEN_CLIP_MODEL = "ViT-L-14-quickgelu"
OPEN_CLIP_PRETRAINED = "dfn2b"
EMBED_DIM = 768
DEFAULT_THRESHOLD = 0.4
DEFAULT_SAMPLE_STEP = 8
DEFAULT_BATCH_SIZE = 16
LOG_INTERVAL_SECS = 10  # in ra progress mỗi 10 giây


# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Logger ghi ra console + file (nếu có)."""
    fmt = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file:
            os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────
# DONE-SET: file ghi các video_id đã xong để resume
# ─────────────────────────────────────────────────────────────────

def load_done_set(done_log_path: str) -> set:
    """Đọc file log done để resume."""
    done = set()
    if os.path.exists(done_log_path):
        with open(done_log_path, "r") as f:
            for line in f:
                vid = line.strip()
                if vid:
                    done.add(vid)
    return done


def mark_done(done_log_path: str, video_id: str, lock: mp.Lock) -> None:
    """Ghi video_id vào done log (thread/process safe)."""
    with lock:
        with open(done_log_path, "a") as f:
            f.write(video_id + "\n")


# ─────────────────────────────────────────────────────────────────
# METADATA HELPERS
# ─────────────────────────────────────────────────────────────────

def load_videos_fps(videos_meta_path: str) -> Dict[str, float]:
    """Đọc videos_metadata.csv → {video_id: fps}."""
    fps_map = {}
    if not os.path.exists(videos_meta_path):
        return fps_map
    with open(videos_meta_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fps_map[row["video_id"]] = float(row["fps"])
            except (KeyError, ValueError):
                pass
    return fps_map


def load_shots_csv(shots_csv_path: str) -> List[Dict]:
    """Đọc *_shots.csv → list of shot dicts."""
    shots = []
    with open(shots_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                shots.append({
                    "shot_id": row["shot_id"],
                    "video_id": row["video_id"],
                    "start_frame": int(float(row["start_frame"])),
                    "end_frame": int(float(row["end_frame"])),
                })
            except (KeyError, ValueError):
                pass
    return shots


def save_keyframes_csv(keyframes: List[Dict], out_path: str) -> None:
    """Lưu keyframes metadata ra CSV."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["keyframe_id", "shot_id", "video_id", "frame_idx", "pts_time", "image_path"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keyframes)


def compare_keyframes_old_vs_new(
    video_id: str,
    new_csv_path: str,
    old_meta_dir: str,
    tol_frames: int = 5,
) -> Optional[Dict]:
    """
    So sánh keyframes mới và cũ cho 1 video_id.
    """
    old_csv_path = os.path.join(old_meta_dir, f"{video_id}_keyframes.csv")
    if not os.path.exists(old_csv_path) or not os.path.exists(new_csv_path):
        return None

    def read_frame_indices(path):
        indices = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    indices.append(int(float(row["frame_idx"])))
                except (KeyError, ValueError, TypeError):
                    pass
        return sorted(indices)

    old_frames = read_frame_indices(old_csv_path)
    new_frames = read_frame_indices(new_csv_path)

    n_old = len(old_frames)
    n_new = len(new_frames)

    if n_old == 0 and n_new == 0:
        return {
            "video_id": video_id,
            "n_old": 0,
            "n_new": 0,
            "n_exact": 0,
            "n_matched_old": 0,
            "n_matched_new": 0,
            "overlap_old_pct": 0.0,
            "overlap_new_pct": 0.0,
        }

    # Match exact
    set_old = set(old_frames)
    set_new = set(new_frames)
    n_exact = len(set_old.intersection(set_new))

    # Match với dung sai tol_frames
    matched_new = sum(
        1 for nf in new_frames if any(abs(nf - of) <= tol_frames for of in old_frames)
    )
    matched_old = sum(
        1 for of in old_frames if any(abs(of - nf) <= tol_frames for nf in new_frames)
    )

    overlap_old_pct = (matched_old / n_old * 100.0) if n_old > 0 else 0.0
    overlap_new_pct = (matched_new / n_new * 100.0) if n_new > 0 else 0.0

    return {
        "video_id": video_id,
        "n_old": n_old,
        "n_new": n_new,
        "n_exact": n_exact,
        "n_matched_old": matched_old,
        "n_matched_new": matched_new,
        "overlap_old_pct": overlap_old_pct,
        "overlap_new_pct": overlap_new_pct,
    }


def print_comparison_summary(results: List[Dict], logger: logging.Logger, tol_frames: int = 5) -> None:
    """In bảng tổng kết so sánh keyframes cũ vs mới."""
    if not results:
        logger.warning("Không có dữ liệu so sánh keyframe nào.")
        return

    logger.info("=" * 105)
    logger.info(f"BÁO CÁO SO SÁNH KEYFRAMES MỚI VS CŨ (Dung sai: ±{tol_frames} frames)")
    logger.info("=" * 105)
    header = f"{'VIDEO ID':<15} | {'CŨ (OLD)':<10} | {'MỚI (NEW)':<10} | {'CHÊNH LỆCH':<10} | {'EXACT':<8} | {'TRÙNG SO CŨ (%)':<20} | {'TRÙNG SO MỚI (%)':<20}"
    logger.info(header)
    logger.info("-" * 105)

    tot_old = 0
    tot_new = 0
    tot_exact = 0
    tot_matched_old = 0
    tot_matched_new = 0

    for r in results:
        v_id = r["video_id"]
        n_old = r["n_old"]
        n_new = r["n_new"]
        n_exact = r["n_exact"]
        m_old = r["n_matched_old"]
        m_new = r["n_matched_new"]
        pct_old = r["overlap_old_pct"]
        pct_new = r["overlap_new_pct"]

        diff = n_new - n_old
        diff_str = f"{diff:+d}"

        tot_old += n_old
        tot_new += n_new
        tot_exact += n_exact
        tot_matched_old += m_old
        tot_matched_new += m_new

        logger.info(
            f"{v_id:<15} | {n_old:<10} | {n_new:<10} | {diff_str:<10} | {n_exact:<8} | {m_old:<4} ({pct_old:5.1f}%)         | {m_new:<4} ({pct_new:5.1f}%)"
        )

    logger.info("-" * 105)
    tot_diff_str = f"{tot_new - tot_old:+d}"
    tot_pct_old = (tot_matched_old / tot_old * 100.0) if tot_old > 0 else 0.0
    tot_pct_new = (tot_matched_new / tot_new * 100.0) if tot_new > 0 else 0.0
    logger.info(
        f"{'TỔNG CỘNG':<15} | {tot_old:<10} | {tot_new:<10} | {tot_diff_str:<10} | {tot_exact:<8} | {tot_matched_old:<4} ({tot_pct_old:5.1f}%)         | {tot_matched_new:<4} ({tot_pct_new:5.1f}%)"
    )
    logger.info("=" * 105)



# ─────────────────────────────────────────────────────────────────
# FRAME READER
# ─────────────────────────────────────────────────────────────────

def read_frame_with_fallback(cap: cv2.VideoCapture, target_frame: int):
    """Đọc frame với fallback ±1 nếu bị lỗi decode."""
    for offset in [0, -1, 1]:
        idx = max(0, target_frame + offset)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            return idx, frame
    return None, None


def collect_shot_candidates(cap: cv2.VideoCapture, start_frame: int, end_frame: int, step: int):
    """Lấy candidate frames trong 1 shot với bước nhảy `step`."""
    candidates = []
    for f in range(start_frame, end_frame + 1, step):
        actual_idx, frame = read_frame_with_fallback(cap, f)
        if frame is not None:
            candidates.append((actual_idx, frame))
    return candidates


# ─────────────────────────────────────────────────────────────────
# OPEN_CLIP EMBEDDER — fixed: L2-normalized output
# ─────────────────────────────────────────────────────────────────

class OpenCLIPEmbedder:
    """
    Wrapper around open_clip ViT-L-14-quickgelu (DFN2B).

    FIX vs code cũ:
    - Dùng open_clip thay vì transformers CLIPModel
    - Trả về L2-normalized embeddings (unit sphere)
    - rel_diff tính trên unit vectors = chỉ đo góc ngữ nghĩa, không bị nhiễu bởi magnitude
    """

    def __init__(self, device: str, batch_size: int = DEFAULT_BATCH_SIZE):
        import open_clip
        from PIL import Image as PILImage

        self.device = device
        self.batch_size = batch_size
        self.PILImage = PILImage

        model, _, preprocess = open_clip.create_model_and_transforms(
            OPEN_CLIP_MODEL,
            pretrained=OPEN_CLIP_PRETRAINED,
            device=device,
        )
        # fp16 để tiết kiệm VRAM trên RTX 2080 Ti
        model = model.half() if "cuda" in device else model
        model.eval()

        self.model = model
        self.preprocess = preprocess

    def encode_images(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        """
        Encode frames BGR → float32 array shape (N, 768), L2-normalized.
        Xử lý theo batch, tự giảm batch_size khi gặp CUDA OOM.
        """
        if not frames_bgr:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)

        chunks = []
        curr_bs = self.batch_size
        idx = 0

        while idx < len(frames_bgr):
            batch_frames = frames_bgr[idx: idx + curr_bs]
            try:
                pil_imgs = []
                for frame in batch_frames:
                    if frame is None or frame.size == 0:
                        blank = np.zeros((224, 224, 3), dtype=np.uint8)
                        pil_imgs.append(self.PILImage.fromarray(blank))
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_imgs.append(self.PILImage.fromarray(rgb))

                # Stack tensor batch
                pixel_values = torch.stack([
                    self.preprocess(img) for img in pil_imgs
                ]).to(self.device)

                if "cuda" in self.device:
                    pixel_values = pixel_values.half()

                with torch.no_grad():
                    features = self.model.encode_image(pixel_values)

                # ─── FIX CHÍNH: L2-normalize về unit sphere ───
                features = F.normalize(features.float(), dim=-1)

                chunks.append(features.cpu().numpy())
                idx += len(batch_frames)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                err = str(exc).lower()
                if ("out of memory" in err or "oom" in err) and curr_bs > 1:
                    new_bs = max(1, curr_bs // 2)
                    logging.warning(
                        "[%s] CUDA OOM @ batch_size=%d → giảm xuống %d",
                        self.device, curr_bs, new_bs,
                    )
                    torch.cuda.empty_cache()
                    curr_bs = new_bs
                else:
                    logging.error(
                        "[%s] encode_image lỗi: %s — dùng zero vector", self.device, exc
                    )
                    chunks.append(np.zeros((len(batch_frames), EMBED_DIM), dtype=np.float32))
                    idx += len(batch_frames)

        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# FILTER — fixed: chạy trên unit-sphere vectors
# ─────────────────────────────────────────────────────────────────

def filter_by_normalized_l2(
    candidates: List[tuple],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[tuple]:
    """
    Lọc keyframes dựa trên L2-norm giữa các unit vectors (đã normalize).

    Công thức:
        rel_diff = ||e_norm_current - e_norm_prev||_2
                 = sqrt(2 * (1 - cosine_similarity))
    → threshold=0.4 tương đương cosine_sim < 0.92

    FIX vs code cũ:
    - Code cũ chia cho norm_prev (không chuẩn khi vector chưa normalize)
    - Code mới: vector đã trên unit sphere, mẫu số = 1, rel_diff = khoảng cách L2 thuần túy
    """
    if not candidates:
        return []

    # Rule 1: Frame đầu tiên của mỗi shot luôn được giữ
    selected = [candidates[0]]
    e_prev = embeddings[0].astype(np.float64)

    for i in range(1, len(candidates)):
        e_curr = embeddings[i].astype(np.float64)

        # L2 distance trên unit sphere = sqrt(2*(1 - cosine_sim))
        rel_diff = np.linalg.norm(e_curr - e_prev)

        if rel_diff > threshold:
            selected.append(candidates[i])
            e_prev = e_curr  # cập nhật mốc so sánh

    return selected


# ─────────────────────────────────────────────────────────────────
# PER-VIDEO EXTRACTOR
# ─────────────────────────────────────────────────────────────────

def extract_keyframes_for_video(
    video_path: str,
    video_id: str,
    shots: List[Dict],
    fps: float,
    embedder: OpenCLIPEmbedder,
    output_dir: str,
    sample_step: int,
    threshold: float,
) -> Optional[List[Dict]]:
    """
    Trích xuất keyframes cho 1 video. Trả về list metadata dicts hoặc None nếu lỗi.
    """
    img_out_dir = os.path.join(output_dir, "images", video_id)
    os.makedirs(img_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Không mở được video: %s", video_path)
        return None

    keyframes = []

    for shot in shots:
        start_frame = shot["start_frame"]
        end_frame = shot["end_frame"]
        shot_id = shot["shot_id"]

        candidates = collect_shot_candidates(cap, start_frame, end_frame, sample_step)
        if not candidates:
            continue

        frames_bgr = [f for _, f in candidates]
        embeddings = embedder.encode_images(frames_bgr)
        selected = filter_by_normalized_l2(candidates, embeddings, threshold)

        for actual_idx, frame in selected:
            kf_num = len(keyframes) + 1
            kf_id = f"{video_id}_kf_{kf_num:04d}"
            img_filename = f"{kf_id}.jpg"
            img_abs_path = os.path.join(img_out_dir, img_filename)
            # image_path tương đối so với output_dir/images/
            img_rel_path = os.path.join("images", video_id, img_filename)

            try:
                ok = cv2.imwrite(img_abs_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    logging.warning("cv2.imwrite thất bại: %s", img_abs_path)
                    continue
            except Exception as exc:
                logging.error("Lỗi ghi ảnh %s: %s", img_abs_path, exc)
                continue

            keyframes.append({
                "keyframe_id": kf_id,
                "shot_id": shot_id,
                "video_id": video_id,
                "frame_idx": actual_idx,
                "pts_time": round(actual_idx / fps, 6) if fps > 0 else 0.0,
                "image_path": img_rel_path,
            })

    cap.release()
    return keyframes


# ─────────────────────────────────────────────────────────────────
# GPU WORKER PROCESS
# ─────────────────────────────────────────────────────────────────

def gpu_worker(
    worker_id: int,
    gpu_id: int,
    video_list: List[Dict],      # [{"video_id", "video_path", "shots", "fps"}, ...]
    output_dir: str,
    sample_step: int,
    threshold: float,
    batch_size: int,
    done_log_path: str,
    done_lock: mp.Lock,
    progress_counter: mp.Value,
    progress_lock: mp.Lock,
    log_file: str,
):
    """
    Process function chạy trên GPU `gpu_id`.
    Xử lý tuần tự từng video trong `video_list`.
    """
    device = f"cuda:{gpu_id}"
    warnings.filterwarnings("ignore")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("open_clip").setLevel(logging.ERROR)

    logger = setup_logger(f"Worker-GPU{gpu_id}", log_file=log_file)

    logger.info("GPU %d | Worker %d khởi động | %d videos", gpu_id, worker_id, len(video_list))

    # Load model 1 lần duy nhất cho cả worker
    try:
        with SuppressCStderr():
            embedder = OpenCLIPEmbedder(device=device, batch_size=batch_size)
        logger.info("GPU %d | OpenCLIP ViT-L-14-quickgelu (DFN2B) loaded OK", gpu_id)
    except Exception as exc:
        logger.error("GPU %d | Không load được model: %s", gpu_id, exc)
        return

    meta_out_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_out_dir, exist_ok=True)

    done_set = load_done_set(done_log_path)

    for entry in video_list:
        video_id = entry["video_id"]
        video_path = entry["video_path"]
        shots = entry["shots"]
        fps = entry["fps"]

        # Resume: skip nếu đã xong
        csv_path = os.path.join(meta_out_dir, f"{video_id}_keyframes.csv")
        if video_id in done_set or os.path.exists(csv_path):
            with progress_lock:
                progress_counter.value += 1
                done_now = progress_counter.value
            logger.info("GPU %d | SKIP (đã xong): %s [%d/%d tổng]", gpu_id, video_id, done_now, total)
            continue

        logger.info("GPU %d | Bắt đầu: %s (%d shots, fps=%.1f)", gpu_id, video_id, len(shots), fps)
        t0 = time.time()

        try:
            with SuppressCStderr():
                keyframes = extract_keyframes_for_video(
                    video_path=video_path,
                    video_id=video_id,
                    shots=shots,
                    fps=fps,
                    embedder=embedder,
                    output_dir=output_dir,
                    sample_step=sample_step,
                    threshold=threshold,
                )
        except Exception as exc:
            with progress_lock:
                progress_counter.value += 1
                done_now = progress_counter.value
            logger.error("GPU %d | LỖI video %s: %s [%d/%d tổng]", gpu_id, video_id, exc, done_now, total)
            continue

        elapsed = time.time() - t0

        if keyframes is None:
            logger.warning("GPU %d | %s → None (không mở được video)", gpu_id, video_id)
        elif not keyframes:
            logger.warning("GPU %d | %s → 0 keyframes", gpu_id, video_id)
        else:
            save_keyframes_csv(keyframes, csv_path)

        # Đánh dấu done (kể cả 0 kf hay lỗi mở video — không retry vô hạn)
        mark_done(done_log_path, video_id, done_lock)

        with progress_lock:
            progress_counter.value += 1
            done_now = progress_counter.value

        kf_count = len(keyframes) if keyframes else 0
        logger.info(
            "GPU %d | ✓ XONG: %s | %d keyframes | %.1fs | [%d/%d tổng đã xong]",
            gpu_id, video_id, kf_count, elapsed, done_now, total,
        )

    logger.info("GPU %d | Worker %d hoàn thành %d videos", gpu_id, worker_id, len(video_list))
    # Giải phóng VRAM
    del embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────
# PROGRESS MONITOR — chạy ở main process
# ─────────────────────────────────────────────────────────────────

def progress_monitor(
    progress_counter: mp.Value,
    progress_lock: mp.Lock,
    total: int,
    stop_event: mp.Event,
    log_file: str,
):
    """In tiến độ ra console & log mỗi LOG_INTERVAL_SECS giây."""
    logger = setup_logger("Progress", log_file=log_file)
    start_time = time.time()
    last_printed = -1

    while not stop_event.is_set():
        time.sleep(LOG_INTERVAL_SECS)
        with progress_lock:
            done = progress_counter.value

        if done != last_printed:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta_secs = (total - done) / rate if rate > 0 else float("inf")
            eta_str = f"{eta_secs/60:.1f} phút" if eta_secs < float("inf") else "N/A"
            logger.info(
                "═══ PROGRESS: %d / %d videos xong (%.1f%%) | tốc độ %.2f vid/phút | ETA: %s ═══",
                done, total, 100.0 * done / total if total else 0, rate * 60, eta_str,
            )
            last_printed = done

    # In lần cuối
    with progress_lock:
        done = progress_counter.value
    logger.info("═══ HOÀN THÀNH: %d / %d videos ═══", done, total)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Regen keyframes với OpenCLIP DFN2B + L2-norm fix, multi-GPU"
    )
    parser.add_argument(
        "--video_dirs", type=str,
        default="/mlcv1/Datasets/HCMAI25/full:/AIClub_NAS/core_baotg/nhan/dataset",
        help="Thư mục chứa *.mp4, ngăn bởi ':', scan recursive. Vd: /dir1:/dir2",
    )
    parser.add_argument(
        "--prefix", type=str, choices=["L", "K"], default="L",
        help="Lọc loại video: 'L' (chạy video loại L, mặc định)",
    )
    parser.add_argument(
        "--shots_dir", type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/shots",
        help="Thư mục chứa *_shots.csv",
    )
    parser.add_argument(
        "--videos_meta", type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos/videos_metadata.csv",
        help="File videos_metadata.csv (để lấy fps)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/keyframes_for_final",
        help="Thư mục đầu ra",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="GPU IDs cách nhau bằng dấu phẩy, vd: 1,2,3,5,6",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Frames/batch khi encode CLIP (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"rel_diff threshold (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--sample_step", type=int, default=DEFAULT_SAMPLE_STEP,
        help=f"Lấy candidate mỗi N frame (default {DEFAULT_SAMPLE_STEP})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Chạy lại kể cả video đã có output",
    )
    parser.add_argument(
        "--video_limit", type=int, default=None,
        help="Giới hạn số video xử lý (debug)",
    )
    parser.add_argument(
        "--old_meta_dir", type=str,
        default="/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes",
        help="Thư mục chứa keyframes metadata cũ để so sánh",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Bật so sánh keyframes mới với keyframes cũ trong old_meta_dir",
    )
    parser.add_argument(
        "--tol_frames", type=int, default=5,
        help="Dung sai frame khi tính trùng giữa keyframe mới và cũ (default 5)",
    )
    return parser.parse_args()


def run_pipeline_for_prefix(prefix_name: str, base_output_dir: str, args):
    """Hàm thực thi pipeline cho một nhóm prefix ('L' hoặc 'K')."""
    target_output_dir = os.path.join(base_output_dir, prefix_name)
    os.makedirs(target_output_dir, exist_ok=True)
    os.makedirs(os.path.join(target_output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_output_dir, "metadata"), exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(target_output_dir, f"regen_keyframes_{prefix_name}_{ts}.log")
    done_log = os.path.join(target_output_dir, "done_videos.log")

    logger = setup_logger(f"Main_{prefix_name}", log_file=log_file)
    logger.info("=" * 70)
    logger.info("regen_keyframes_final.py START [PREFIX %s] — %s", prefix_name, ts)
    logger.info("Output dir  : %s", target_output_dir)
    logger.info("Video dirs  : %s", args.video_dirs)
    logger.info("Shots dir   : %s", args.shots_dir)
    logger.info("GPUs        : %s", args.gpus)
    logger.info("Threshold   : %.2f", args.threshold)
    logger.info("Sample step : %d", args.sample_step)
    logger.info("Batch size  : %d", args.batch_size)
    logger.info("Force rerun : %s", args.force)
    logger.info("=" * 70)

    # ── Parse GPU list ──
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        logger.error("Không có GPU nào được chỉ định.")
        return

    # ── Load fps map ──
    fps_map = load_videos_fps(args.videos_meta)

    # ── Quét shots files theo Prefix (ví dụ: L*_shots.csv hoặc K*_shots.csv) ──
    shot_pattern = f"{prefix_name}*_shots.csv" if prefix_name in ["L", "K"] else "*_shots.csv"
    shot_csvs = sorted(glob.glob(os.path.join(args.shots_dir, shot_pattern)))
    logger.info("Tìm thấy %d file shots CSV khớp pattern '%s'", len(shot_csvs), shot_pattern)

    # ── Scan video_dirs ──
    video_dirs = [d.strip() for d in args.video_dirs.split(":") if d.strip()]
    video_path_map: Dict[str, str] = {}
    for vd in video_dirs:
        if not os.path.isdir(vd):
            continue
        found = glob.glob(os.path.join(vd, "**", f"{prefix_name}*.mp4" if prefix_name in ["L", "K"] else "*.mp4"), recursive=True)
        for fpath in found:
            vid = os.path.splitext(os.path.basename(fpath))[0]
            if vid not in video_path_map:
                video_path_map[vid] = fpath

    logger.info("Tổng mp4 tìm thấy cho prefix %s: %d files", prefix_name, len(video_path_map))

    if args.force and os.path.exists(done_log):
        logger.warning("--force: xóa done_videos.log [%s], chạy lại từ đầu", prefix_name)
        os.remove(done_log)

    all_videos = []
    skipped_no_video = 0
    skipped_no_shot = 0

    for shot_csv in shot_csvs:
        fname = os.path.basename(shot_csv)
        video_id = fname.replace("_shots.csv", "")

        # Lọc lại chính xác theo prefix
        if prefix_name in ["L", "K"] and not video_id.startswith(prefix_name):
            continue

        video_path = video_path_map.get(video_id)
        if not video_path:
            skipped_no_video += 1
            continue

        shots = load_shots_csv(shot_csv)
        if not shots:
            skipped_no_shot += 1
            continue

        fps = fps_map.get(video_id, 0.0)
        if fps == 0.0:
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
            except Exception:
                fps = 30.0

        all_videos.append({
            "video_id": video_id,
            "video_path": video_path,
            "shots": shots,
            "fps": fps,
        })

    logger.info(
        "[%s] Tổng video hợp lệ: %d | bỏ qua (không tìm thấy mp4): %d | bỏ qua (shot rỗng): %d",
        prefix_name, len(all_videos), skipped_no_video, skipped_no_shot,
    )

    if args.video_limit:
        all_videos = all_videos[: args.video_limit]

    total = len(all_videos)
    if total == 0:
        logger.warning("[%s] Không có video nào để xử lý.", prefix_name)
        return

    # ── Phân chia video ──
    n_workers = len(gpu_ids)
    chunks: List[List[Dict]] = [[] for _ in range(n_workers)]
    for i, v in enumerate(all_videos):
        chunks[i % n_workers].append(v)

    for i, gid in enumerate(gpu_ids):
        logger.info("[%s] GPU %d → %d videos", prefix_name, gid, len(chunks[i]))

    ctx = mp.get_context("spawn")
    progress_counter = ctx.Value("i", 0)
    progress_lock = ctx.Lock()
    done_lock = ctx.Lock()
    stop_event = ctx.Event()

    monitor_proc = ctx.Process(
        target=progress_monitor,
        args=(progress_counter, progress_lock, total, stop_event, log_file),
        daemon=True,
    )
    monitor_proc.start()

    workers = []
    for worker_id, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
        if not chunk:
            continue
        p = ctx.Process(
            target=gpu_worker,
            kwargs=dict(
                worker_id=worker_id,
                gpu_id=gpu_id,
                video_list=chunk,
                output_dir=target_output_dir,
                sample_step=args.sample_step,
                threshold=args.threshold,
                batch_size=args.batch_size,
                done_log_path=done_log,
                done_lock=done_lock,
                progress_counter=progress_counter,
                progress_lock=progress_lock,
                log_file=log_file,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    stop_event.set()
    monitor_proc.join(timeout=30)

    with progress_lock:
        final_done = progress_counter.value
    logger.info("=" * 70)
    logger.info("[%s] TỔNG KẾT: %d / %d videos đã xử lý", prefix_name, final_done, total)
    logger.info("[%s] Output: %s", prefix_name, target_output_dir)
    logger.info("=" * 70)

    # ── So sánh Keyframe Mới vs Cũ (nếu bật --compare hoặc có --video_limit) ──
    if args.compare or args.video_limit is not None:
        logger.info("Đang tính toán so sánh keyframes mới vs cũ...")
        comp_results = []
        meta_out_dir = os.path.join(target_output_dir, "metadata")
        for entry in all_videos:
            vid = entry["video_id"]
            new_csv = os.path.join(meta_out_dir, f"{vid}_keyframes.csv")
            res = compare_keyframes_old_vs_new(
                video_id=vid,
                new_csv_path=new_csv,
                old_meta_dir=args.old_meta_dir,
                tol_frames=args.tol_frames,
            )
            if res:
                comp_results.append(res)
        print_comparison_summary(comp_results, logger=logger, tol_frames=args.tol_frames)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("LỖI: CUDA không khả dụng!")
        sys.exit(1)

    print(f">>> BẮT ĐẦU TRÍCH XUẤT KEYFRAMES CHO VIDEO LOẠI [{args.prefix}] <<<")
    run_pipeline_for_prefix(args.prefix, args.output_dir, args)


if __name__ == "__main__":
    # Cần "spawn" context để tránh lỗi CUDA + fork
    mp.set_start_method("spawn", force=True)
    main()
