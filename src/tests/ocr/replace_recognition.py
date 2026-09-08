#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replace_recognition.py
=======================

Pipeline:
    OCR.json (PaddleOCR, đã có polygon)
        --> merge box cùng dòng
        --> tính box_width
        --> box_width < width_threshold  => giữ nguyên text PaddleOCR
            box_width >= width_threshold => crop polygon + VietOCR batch recognize
        --> OCR_vietocr.json

    Lý do lọc theo width: box hẹp (watermark, logo kênh, chữ nhỏ góc màn
    hình...) thường bị VietOCR đọc sai hoàn toàn (VD PaddleOCR đọc đúng
    "HTV9D" nhưng VietOCR đọc thành "Chillier"), trong khi VietOCR lại chính
    xác hơn PaddleOCR trên các dòng chữ dài (tiêu đề, phụ đề). Threshold mặc
    định 150px, chỉnh bằng --width_threshold.

Mỗi file input có dạng:
[
  {
    "doc_id": 0,
    "image": "/path/to/keyframe.jpg",
    "width": 1280,
    "height": 720,
    "texts": [
        {"text": "HTV9P", "score": 0.89, "polygon": [[x,y], [x,y], [x,y], [x,y]]},
        ...
    ],
    "text": "HTV9P 06:30:11 HD giây"
  },
  ...
]

Script sẽ:
  1. Đọc từng file .json trong --ocr_dir (theo --pattern).
  2. Với mỗi doc, đọc ảnh gốc (image path trong json), có thể remap
     lại root path bằng --image_root_replace nếu chạy trên máy khác NAS.
  3. Group các polygon cùng dòng và merge thành một bounding rectangle.
     (Có thể tắt bước này bằng --no_merge để dùng polygon gốc.)
  4. Tính box_width của mỗi box (sau merge). Box hẹp hơn --width_threshold
     (mặc định 150px) giữ nguyên text PaddleOCR (field "recognized_by" =
     "paddle"). Box đủ rộng mới crop vùng polygon và gom vào batch để đưa
     qua VietOCR (field "recognized_by" = "vietocr").
  5. Khi đủ batch_size (hoặc hết doc), gọi VietOCR predict_batch 1 lần.
  6. Ghi kết quả nhận dạng lại vào field mới "text_vietocr" (mặc định,
     không đụng field "text" gốc của PaddleOCR để dễ so sánh), hoặc ghi
     đè trực tiếp lên field "text" nếu dùng --inplace.
  7. Ghi ra --output_dir với tên file giữ nguyên (VD: L21_V001.json).

Cài đặt cần thiết:
    pip install vietocr opencv-python pillow tqdm

Ví dụ chạy:
    python replace_recognition.py \
        --ocr_dir /AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/ocr \
        --output_dir /AIClub_NAS/core_baotg/nhan/data-AIC2025/outputs/ocr_vietocr \
        --device cuda:0 \
        --batch_size 64

Nếu chạy trên máy không mount đúng "/AIClub_NAS/..." mà mount ở chỗ
khác, dùng --image_root_replace để remap:
    --image_root_replace /AIClub_NAS/core_baotg/nhan/dataset:/mnt/dataset

Chạy song song nhiều tiến trình (VD nhiều tmux) trỏ chung 1 --output_dir:
    Script tự tạo file lock "<tên_file>.json.lock" trong output_dir trước khi
    xử lý mỗi file. Tiến trình nào tạo lock trước thì xử lý, các tiến trình
    khác thấy lock đã tồn tại sẽ tự bỏ qua file đó và chuyển sang file tiếp
    theo - nên chỉ cần chạy cùng 1 lệnh giống hệt nhau ở mỗi tmux là được,
    không cần chia file thủ công. Lock được xoá ngay sau khi xử lý xong (dù
    thành công, lỗi, hay bị Ctrl+C). Nếu 1 tiến trình bị kill cứng (kill -9,
    mất kết nối SSH đột ngột...) mà không kịp dọn lock, sau --lock_timeout
    giây (mặc định 3600s) tiến trình khác sẽ tự coi lock đó là "chết" và
    chiếm lại. Dùng --no_lock nếu chắc chắn chỉ chạy 1 tiến trình duy nhất.
"""

import os
import sys
import json
import glob
import time
import socket
import argparse
import logging
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    print(
        "[!] Chưa cài vietocr. Cài bằng: pip install vietocr",
        file=sys.stderr,
    )
    raise


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("replace_recognition")


# --------------------------------------------------------------------------- #
# VietOCR predictor
# --------------------------------------------------------------------------- #
def build_predictor(config_name: str, weights: Optional[str], device: str) -> Predictor:
    """Khởi tạo VietOCR Predictor."""
    config = Cfg.load_config_from_name(config_name)
    if weights:
        config["weights"] = weights
    config["cnn"]["pretrained"] = False
    config["device"] = device
    config["predictor"]["beamsearch"] = False
    log.info(f"Load VietOCR config='{config_name}' device='{device}' weights='{weights or 'default'}'")
    return Predictor(config)


def _is_oom_error(e: Exception) -> bool:
    """Nhận diện lỗi hết VRAM (CUDA OOM), để phân biệt với các lỗi khác (input
    hỏng, lỗi decode...) - chỉ lỗi OOM mới đáng để chia nhỏ batch thử lại,
    lỗi khác thì chia nhỏ cũng không giải quyết được gì."""
    try:
        import torch
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class BatchSizer:
    """Theo dõi kích thước batch VietOCR còn "chạy được" trên GPU hiện tại,
    tự giảm dần khi gặp OOM (64 -> 32 -> 16 -> 8 -> ...) và GHI NHỚ mức đó
    cho các lần gọi tiếp theo trong cùng lần chạy file - tránh phải dò lại
    từ 64 mỗi lần flush() một khi đã biết batch lớn không chạy được. Quan
    trọng với file có hàng chục nghìn vùng text: nếu không nhớ lại, mỗi
    lần flush() sẽ tốn thêm vài lần predict thất bại (64->32->16) trước khi
    mới predict thành công ở 8 - rất phí.
    """

    def __init__(self, initial: int):
        self.size = max(1, initial)

    def shrink_to(self, new_size: int) -> None:
        new_size = max(1, new_size)
        if new_size < self.size:
            self.size = new_size


def recognize_batch(
    predictor: Predictor,
    pil_images: List[Image.Image],
    sizer: Optional[BatchSizer] = None,
    _depth: int = 0,
) -> List[str]:
    """
    Nhận dạng 1 batch ảnh PIL bằng VietOCR.

    Nếu gặp OOM: KHÔNG rơi thẳng xuống predict từng ảnh 1 (rất phí nếu file
    có hàng chục nghìn vùng text - mất hẳn lợi ích chạy batch trên GPU), mà
    CHIA ĐÔI batch và thử lại đệ quy: 64 -> 32 -> 16 -> 8 -> ... cho tới khi
    tìm được kích thước chạy được. Chỉ khi batch=1 mà vẫn OOM (1 ảnh đã quá
    lớn so với VRAM còn trống) mới thực sự predict đơn lẻ / trả về rỗng.

    sizer (tuỳ chọn): nếu truyền vào, kích thước batch "an toàn" học được
    sẽ được ghi lại vào sizer.size để nơi gọi (process_file) dùng cho các
    batch tiếp theo, không phải dò lại từ đầu mỗi lần.

    Lỗi KHÔNG PHẢI OOM (VD ảnh lỗi decode, input sai định dạng...) thì chia
    nhỏ không giải quyết được gì - fallback predict từng ảnh 1 lần cho đúng
    batch hiện tại, không đệ quy chia tiếp.
    """
    if not pil_images:
        return []

    # Nếu đã biết trước (từ lần OOM trước đó trong sizer) rằng batch này chắc
    # chắn vượt mức an toàn, chia thẳng theo mức đã biết thay vì tốn 1 lần
    # gọi predict_batch chắc chắn thất bại rồi mới chia đôi từ đầu.
    if sizer is not None and len(pil_images) > sizer.size:
        known_size = sizer.size
        results: List[str] = []
        for start in range(0, len(pil_images), known_size):
            chunk = pil_images[start:start + known_size]
            results.extend(recognize_batch(predictor, chunk, sizer, _depth + 1))
        return results

    try:
        return predictor.predict_batch(pil_images)
    except AttributeError:
        # bản vietocr cũ không có predict_batch -> đành predict từng ảnh
        return [predictor.predict(img) for img in pil_images]
    except Exception as e:
        n = len(pil_images)

        if n == 1:
            if _is_oom_error(e):
                log.error(f"OOM ngay cả với batch=1 (1 ảnh), có thể ảnh quá lớn: {e}")
                _clear_cuda_cache()
            else:
                log.warning(f"Predict 1 ảnh lỗi: {e}")
            return [""]

        if not _is_oom_error(e):
            log.warning(
                f"Batch predict lỗi không phải OOM (batch={n}): {e} "
                f"-> fallback predict từng ảnh trong batch này (không chia tiếp)."
            )
            results = []
            for img in pil_images:
                try:
                    results.append(predictor.predict(img))
                except Exception as e2:
                    log.warning(f"Predict 1 ảnh cũng lỗi: {e2}")
                    results.append("")
            return results

        # OOM thật sự -> chia đôi, thử lại đệ quy cho từng nửa
        mid = n // 2
        log.warning(f"OOM với batch={n} (depth={_depth}) -> giải phóng cache, chia đôi ({mid} + {n - mid}) và thử lại.")
        _clear_cuda_cache()
        if sizer is not None:
            sizer.shrink_to(mid)
        left = recognize_batch(predictor, pil_images[:mid], sizer, _depth + 1)
        right = recognize_batch(predictor, pil_images[mid:], sizer, _depth + 1)
        return left + right


# --------------------------------------------------------------------------- #
# Crop theo polygon
# --------------------------------------------------------------------------- #
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm polygon thành thứ tự tl, tr, br, bl."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_polygon(image: np.ndarray, polygon, pad: int = 2) -> Optional[np.ndarray]:
    """
    Crop vùng tứ giác (polygon) từ ảnh bằng perspective warp, hỗ trợ box
    bị nghiêng (không phải hình chữ nhật thẳng trục).
    - image: ảnh numpy RGB (H, W, 3)
    - polygon: list 4 điểm [[x,y], [x,y], [x,y], [x,y]]
    Trả về ảnh đã crop (numpy RGB) hoặc None nếu polygon không hợp lệ.
    """
    h_img, w_img = image.shape[:2]
    pts = np.array(polygon, dtype=np.float32)

    if pts.shape[0] != 4:
        # polygon không phải tứ giác (VD >4 điểm) -> fallback bounding rect
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        x = max(0, x - pad)
        y = max(0, y - pad)
        x2 = min(w_img, x + w + 2 * pad)
        y2 = min(h_img, y + h + 2 * pad)
        crop = image[y:y2, x:x2]
        return crop if crop.size > 0 else None

    # clip điểm nằm trong ảnh để tránh lỗi warp do polygon vượt biên
    pts[:, 0] = np.clip(pts[:, 0], 0, w_img - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h_img - 1)

    ordered = order_quad_points(pts)
    tl, tr, br, bl = ordered

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(round(max(width_a, width_b)))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(round(max(height_a, height_b)))

    if max_width < 2 or max_height < 2:
        return None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    try:
        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
    except cv2.error as e:
        log.warning(f"warpPerspective lỗi: {e}")
        return None

    # nếu box dạng chữ dọc (cao hơn rộng nhiều) -> xoay ngang cho VietOCR
    if warped.shape[0] > warped.shape[1] * 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped


# --------------------------------------------------------------------------- #
# Merge các polygon bị fragment (cùng dòng) trước khi recognition
# --------------------------------------------------------------------------- #
def _polygon_bbox(polygon) -> Optional[Tuple[float, float, float, float]]:
    """Trả về x1, y1, x2, y2 của polygon hoặc None nếu polygon không hợp lệ."""
    try:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            return None
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        if x2 <= x1 or y2 <= y1:
            return None
        return float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        return None


def _should_merge(
    abox: Tuple[float, float, float, float],
    bbox: Tuple[float, float, float, float],
    same_line_y: float,
    max_gap: float,
    min_height_ratio: float,
) -> bool:
    """Hai box chỉ merge nếu cùng dòng, cao tương tự và overlap/gần nhau."""
    ax1, ay1, ax2, ay2 = abox
    bx1, by1, bx2, by2 = bbox
    ah, bh = ay2 - ay1, by2 - by1
    acy, bcy = (ay1 + ay2) / 2, (by1 + by2) / 2
    same_line = abs(acy - bcy) < same_line_y
    similar_height = min(ah, bh) / max(ah, bh) > min_height_ratio
    x_gap = max(bx1 - ax2, ax1 - bx2, 0.0)
    horizontal_overlap = min(ax2, bx2) >= max(ax1, bx1)
    return same_line and similar_height and (horizontal_overlap or x_gap < max_gap)


def merge_text_items(
    items: List[dict],
    min_score: float,
    same_line_y: float = 0.4,
    max_gap: float = 1.5,
    min_height_ratio: float = 0.6,
    absolute_thresholds: bool = False,
) -> List[dict]:
    """Merge các box detect vỡ cùng 1 dòng bằng connected components hình học.

    Chỉ các item có polygon và đạt min_score được xét merge. Item bị bỏ qua bởi
    score filter hoặc không có polygon vẫn được giữ nguyên (không đổi) trong output.

    same_line_y / max_gap: theo mặc định là HỆ SỐ nhân với chiều cao trung bình
    của 2 box đang xét (scale theo cỡ chữ), không phải pixel tuyệt đối - để
    không bị lệch khi trong cùng video có cả chữ to (tiêu đề) lẫn chữ nhỏ
    (đồng hồ góc màn hình, phụ đề...). Truyền absolute_thresholds=True nếu
    muốn dùng thẳng 2 giá trị này như pixel cố định (hành vi cũ).

    Lưu ý quan trọng: quan hệ "nên merge" giữa 2 box là quan hệ xấp xỉ, không
    có tính bắc cầu (A gần B, B gần C không đảm bảo A gần C). Union-Find gộp
    theo bắc cầu nên 1 nhóm có thể chứa các box ở 2 đầu cách nhau xa hơn
    ngưỡng đặt ra. Đây là đánh đổi chấp nhận được để xử lý dòng dài bị vỡ
    thành >2 mảnh, nhưng cần biết để không ngạc nhiên khi thấy nhóm merge
    "hơi rộng" hơn ngưỡng nhìn thoáng qua.
    """
    candidates = []
    untouched = []
    for index, item in enumerate(items):
        polygon = item.get("polygon")
        if not polygon or (min_score > 0 and item.get("score", 1.0) < min_score):
            untouched.append((index, item))
            continue
        bbox = _polygon_bbox(polygon)
        if bbox is None:
            untouched.append((index, item))
            continue
        candidates.append((index, bbox, item))

    parent = list(range(len(candidates)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(candidates)):
        _, abox, _ = candidates[i]
        ah = abox[3] - abox[1]
        for j in range(i + 1, len(candidates)):
            _, bbox, _ = candidates[j]
            bh = bbox[3] - bbox[1]
            if absolute_thresholds:
                thr_y, thr_gap = same_line_y, max_gap
            else:
                avg_h = (ah + bh) / 2.0
                thr_y = same_line_y * avg_h
                thr_gap = max_gap * avg_h
            if _should_merge(abox, bbox, thr_y, thr_gap, min_height_ratio):
                union(i, j)

    groups = {}
    for i, candidate in enumerate(candidates):
        groups.setdefault(find(i), []).append(candidate)

    merged = []
    for group in groups.values():
        if len(group) == 1:
            idx, _, item = group[0]
            merged.append((idx, item))
            continue

        # QUAN TRỌNG: sắp theo toạ độ x trái (bbox[0]) - tức thứ tự đọc trái->phải
        # trên ảnh - KHÔNG sắp theo index detect gốc của PaddleOCR, vì thứ tự
        # PaddleOCR trả về không đảm bảo đúng thứ tự đọc trong 1 dòng. Sắp sai
        # sẽ làm field "text" (ghép từ PaddleOCR) bị đảo từ, dù "text_vietocr"
        # (đọc thẳng từ ảnh crop) vẫn không bị ảnh hưởng.
        group.sort(key=lambda entry: entry[1][0])

        first_index = min(entry[0] for entry in group)
        boxes = [entry[1] for entry in group]
        source_items = [entry[2] for entry in group]

        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        merged_item = dict(source_items[0])
        merged_item["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        merged_item["text"] = " ".join(
            item.get("text", "").strip() for item in source_items if item.get("text", "").strip()
        )
        merged_item["score"] = min(item.get("score", 1.0) for item in source_items)
        merged_item["merged_count"] = len(source_items)
        merged.append((first_index, merged_item))

    merged.extend(untouched)
    merged.sort(key=lambda x: x[0])
    return [item for _, item in merged]


# --------------------------------------------------------------------------- #
# Đọc / remap đường dẫn ảnh
# --------------------------------------------------------------------------- #
def parse_root_replace(pairs: List[str]) -> Dict[str, str]:
    """Parse list "old:new" -> dict {old: new}."""
    mapping = {}
    for p in pairs:
        if ":" not in p:
            log.warning(f"Bỏ qua --image_root_replace không hợp lệ: {p}")
            continue
        old, new = p.split(":", 1)
        mapping[old] = new
    return mapping


def remap_path(path: str, root_map: Dict[str, str]) -> str:
    for old, new in root_map.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_image_rgb(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# File lock - cho phép nhiều tiến trình (VD nhiều tmux) cùng chạy script,
# trỏ chung 1 --output_dir, mà không xử lý trùng 1 file .json.
# --------------------------------------------------------------------------- #
def _lock_path(output_dir: str, json_path: str) -> str:
    return os.path.join(output_dir, os.path.basename(json_path) + ".lock")


def acquire_lock(lock_path: str, stale_seconds: float, _retried: bool = False) -> bool:
    """
    Cố chiếm lock bằng cách tạo file atomic (O_CREAT|O_EXCL - chỉ 1 tiến trình
    tạo thành công nếu nhiều tiến trình cùng gọi song song, do hệ điều hành
    đảm bảo tính atomic của thao tác này kể cả trên NFS/NAS thường gặp).

    Trả về True nếu chiếm được lock (tiến trình này được xử lý file), False
    nếu file đang bị 1 tiến trình khác giữ lock (nên bỏ qua, để tiến trình
    kia xử lý).

    stale_seconds: nếu lock đã tồn tại quá lâu (tiến trình giữ lock trước đó
    khả năng đã crash/bị kill mà không kịp dọn lock), tự động coi là lock
    "chết" và chiếm lại. Đặt 0 để tắt cơ chế này (lock không bao giờ tự hết
    hạn, phải xoá tay file .lock nếu bị treo do crash).
    """
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": time.time(),
                }))
        except Exception:
            pass
        return True
    except FileExistsError:
        if _retried or stale_seconds <= 0:
            return False
        try:
            age = time.time() - os.path.getmtime(lock_path)
        except OSError:
            age = 0.0
        if age > stale_seconds:
            log.warning(
                f"Lock '{lock_path}' đã tồn tại {age:.0f}s (> {stale_seconds:.0f}s) "
                f"-> coi như tiến trình giữ lock trước đó đã crash, chiếm lại."
            )
            try:
                os.remove(lock_path)
            except OSError:
                pass
            # chỉ thử lại đúng 1 lần để tránh vòng lặp nếu có race giữa nhiều
            # tiến trình cùng phát hiện stale lock cùng lúc
            return acquire_lock(lock_path, stale_seconds, _retried=True)
        return False


def release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Xử lý 1 file OCR.json
# --------------------------------------------------------------------------- #
def process_file(
    json_path: str,
    output_dir: str,
    predictor: Predictor,
    root_map: Dict[str, str],
    batch_size: int,
    min_score: float,
    inplace: bool,
    num_workers: int,
    same_line_y: float,
    max_gap: float,
    min_height_ratio: float,
    no_merge: bool,
    absolute_thresholds: bool,
    width_threshold: float,
) -> Tuple[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # remap + resolve đường dẫn ảnh cho từng doc trước
    image_paths = []
    for doc in data:
        p = remap_path(doc.get("image", ""), root_map)
        image_paths.append(p)

    # prefetch ảnh song song (NAS I/O có thể chậm) theo đúng thứ tự doc
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        image_iter = ex.map(load_image_rgb, image_paths)

        batch_imgs: List[Image.Image] = []
        batch_refs: List[Tuple[int, int]] = []
        n_recognized = 0
        sizer = BatchSizer(batch_size)

        def flush():
            nonlocal batch_imgs, batch_refs, n_recognized
            if not batch_imgs:
                return
            texts = recognize_batch(predictor, batch_imgs, sizer)
            for (d_idx, t_idx), text in zip(batch_refs, texts):
                item = data[d_idx]["texts"][t_idx]
                if inplace:
                    item["text"] = text
                else:
                    item["text_vietocr"] = text
            n_recognized += len(batch_imgs)
            batch_imgs = []
            batch_refs = []

        pbar = tqdm(
            zip(data, image_iter),
            total=len(data),
            desc=os.path.basename(json_path),
            leave=False,
        )
        for d_idx, (doc, img_rgb) in enumerate(pbar):
            if no_merge:
                texts = doc.get("texts", [])
            else:
                texts = merge_text_items(
                    doc.get("texts", []),
                    min_score=min_score,
                    same_line_y=same_line_y,
                    max_gap=max_gap,
                    min_height_ratio=min_height_ratio,
                    absolute_thresholds=absolute_thresholds,
                )
            doc["texts"] = texts
            if img_rgb is None:
                log.warning(f"Không đọc được ảnh: {image_paths[d_idx]} (doc_id={doc.get('doc_id')})")
                for item in texts:
                    fallback = item.get("text", "")
                    item["recognized_by"] = "paddle"
                    if not inplace:
                        item["text_vietocr"] = fallback
                continue

            for t_idx, item in enumerate(texts):
                polygon = item.get("polygon")
                if not polygon:
                    item["recognized_by"] = "paddle"
                    if not inplace:
                        item["text_vietocr"] = item.get("text", "")
                    continue
                if min_score > 0 and item.get("score", 1.0) < min_score:
                    item["recognized_by"] = "paddle"
                    if not inplace:
                        item["text_vietocr"] = item.get("text", "")
                    continue

                # Box (đã merge cùng dòng, nếu có) quá hẹp -> thường là watermark/
                # logo kênh/chữ nhỏ ở góc màn hình, VietOCR hay đọc sai kiểu này
                # (thấy trong thực tế: box hẹp bị nhận nhầm hoàn toàn, VD "HTV9D"
                # -> "Chillier"). Những box này giữ nguyên kết quả PaddleOCR thay
                # vì đưa qua VietOCR. Chỉ box đủ rộng (thường là dòng chữ dài,
                # tiêu đề, phụ đề) mới đưa qua VietOCR để tận dụng độ chính xác
                # cao hơn của nó trên chuỗi dài.
                bbox = _polygon_bbox(polygon)
                box_width = (bbox[2] - bbox[0]) if bbox else 0.0
                if box_width < width_threshold:
                    item["recognized_by"] = "paddle"
                    item["box_width"] = box_width
                    if not inplace:
                        item["text_vietocr"] = item.get("text", "")
                    continue

                crop = crop_polygon(img_rgb, polygon)
                if crop is None or min(crop.shape[0], crop.shape[1]) < 2:
                    item["recognized_by"] = "paddle"
                    if not inplace:
                        item["text_vietocr"] = item.get("text", "")
                    continue

                item["recognized_by"] = "vietocr"
                item["box_width"] = box_width
                batch_imgs.append(Image.fromarray(crop))
                batch_refs.append((d_idx, t_idx))
                if len(batch_imgs) >= sizer.size:
                    flush()

        flush()

    # tính lại field text tổng hợp cho mỗi doc
    for doc in data:
        if inplace:
            doc["text"] = " ".join(t.get("text", "") for t in doc.get("texts", []))
        else:
            doc["text_vietocr"] = " ".join(
                t.get("text_vietocr", t.get("text", "")) for t in doc.get("texts", [])
            )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(json_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return out_path, n_recognized


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Crop theo polygon từ OCR.json (PaddleOCR) và nhận dạng lại bằng VietOCR."
    )
    parser.add_argument("--ocr_dir", required=True, help="Thư mục chứa các file OCR .json (VD: .../outputs/ocr)")
    parser.add_argument("--output_dir", required=True, help="Thư mục ghi kết quả (VD: .../outputs/ocr_vietocr)")
    parser.add_argument("--pattern", default="*.json", help="Pattern chọn file trong ocr_dir (mặc định *.json)")
    parser.add_argument("--files", nargs="*", default=None, help="Chỉ định trực tiếp danh sách file .json cần xử lý (bỏ qua --pattern)")

    parser.add_argument(
        "--config_name",
        default="vgg_transformer",
        choices=["vgg_transformer", "vgg_seq2seq"],
        help="Config kiến trúc VietOCR",
    )
    parser.add_argument("--weights", default=None, help="Đường dẫn file .pth weights VietOCR (mặc định dùng pretrained có sẵn)")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 hoặc cpu")
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Số ảnh crop / lần predict_batch (điểm khởi đầu). Nếu gặp OOM, "
        "script tự chia đôi và thử lại (64->32->16->8->...), rồi GHI NHỚ mức "
        "chạy được đó để dùng cho các batch tiếp theo trong cùng file, thay "
        "vì dò lại từ đầu mỗi lần hoặc rơi thẳng xuống predict từng ảnh 1.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Số thread đọc ảnh song song")

    parser.add_argument(
        "--no_merge", action="store_true",
        help="Tắt bước merge polygon cùng dòng, dùng nguyên polygon gốc từ PaddleOCR",
    )
    parser.add_argument(
        "--same_line_y", type=float, default=0.4,
        help="Ngưỡng lệch tâm y để coi 2 box cùng dòng. Mặc định là HỆ SỐ nhân với "
        "chiều cao trung bình 2 box (VD 0.4 = lệch < 40%% chiều cao chữ). "
        "Dùng --absolute_thresholds để coi đây là pixel cố định.",
    )
    parser.add_argument(
        "--max_gap", type=float, default=1.5,
        help="Khoảng cách ngang tối đa để merge 2 box không overlap. Mặc định là "
        "HỆ SỐ nhân với chiều cao trung bình 2 box. Dùng --absolute_thresholds "
        "để coi đây là pixel cố định.",
    )
    parser.add_argument(
        "--min_height_ratio", type=float, default=0.6,
        help="Tỷ lệ chiều cao nhỏ/lớn tối thiểu khi merge (0-1)",
    )
    parser.add_argument(
        "--absolute_thresholds", action="store_true",
        help="Coi --same_line_y và --max_gap là pixel tuyệt đối thay vì hệ số "
        "nhân theo chiều cao box (hành vi cũ, không khuyến nghị nếu video có "
        "nhiều cỡ chữ khác nhau)",
    )
    parser.add_argument(
        "--width_threshold", type=float, default=150.0,
        help="Chiều rộng box (px, sau khi merge cùng dòng nếu có) tối thiểu để "
        "đưa qua VietOCR nhận dạng lại. Box hẹp hơn (thường là watermark/logo/"
        "chữ nhỏ góc màn hình) sẽ giữ nguyên kết quả PaddleOCR vì VietOCR hay "
        "đọc sai loại này. Đặt 0 để luôn dùng VietOCR cho mọi box.",
    )

    parser.add_argument(
        "--image_root_replace",
        nargs="*",
        default=[],
        help='Remap root path ảnh dạng "old_root:new_root", có thể truyền nhiều cặp. '
        'VD: --image_root_replace /AIClub_NAS/core_baotg/nhan/dataset:/mnt/dataset',
    )
    parser.add_argument("--min_score", type=float, default=0.0, help="Bỏ qua vùng có score detect thấp hơn ngưỡng này (0 = không lọc)")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Ghi đè trực tiếp field 'text' bằng kết quả VietOCR thay vì thêm field mới 'text_vietocr'",
    )
    parser.add_argument("--overwrite", action="store_true", help="Ghi đè file output nếu đã tồn tại (mặc định sẽ skip file đã xử lý)")

    parser.add_argument(
        "--no_lock", action="store_true",
        help="Tắt cơ chế lock. Chỉ dùng khi CHẮC CHẮN chỉ có 1 tiến trình xử lý "
        "chung --output_dir này, nếu không sẽ có 2 tiến trình xử lý trùng file.",
    )
    parser.add_argument(
        "--lock_timeout", type=float, default=3600.0,
        help="Số giây coi lock của 1 tiến trình khác là 'chết' (do crash/bị kill "
        "mà không dọn lock) và tự chiếm lại (mặc định 3600s = 1 giờ). Đặt 0 để "
        "tắt cơ chế này - lock không bao giờ tự hết hạn, phải xoá tay file "
        "'<tên_file>.json.lock' trong output_dir nếu bị treo do crash.",
    )

    args = parser.parse_args()

    # xác định danh sách file cần xử lý
    if args.files:
        json_files = args.files
    else:
        json_files = sorted(glob.glob(os.path.join(args.ocr_dir, args.pattern)))

    if not json_files:
        log.error(f"Không tìm thấy file json nào trong {args.ocr_dir} (pattern={args.pattern})")
        sys.exit(1)

    root_map = parse_root_replace(args.image_root_replace)

    predictor = build_predictor(args.config_name, args.weights, args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    total_recognized = 0
    total_files_done = 0
    total_skipped_locked = 0
    for jp in tqdm(json_files, desc="Files"):
        out_path = os.path.join(args.output_dir, os.path.basename(jp))
        if os.path.exists(out_path) and not args.overwrite:
            log.info(f"Skip (đã có output): {out_path}")
            continue

        lock_path = _lock_path(args.output_dir, jp)
        if not args.no_lock:
            if not acquire_lock(lock_path, args.lock_timeout):
                log.info(f"Skip (đang bị tiến trình khác xử lý, xem lock={lock_path}): {jp}")
                total_skipped_locked += 1
                continue

        try:
            out_path, n_rec = process_file(
                jp,
                args.output_dir,
                predictor,
                root_map,
                args.batch_size,
                args.min_score,
                args.inplace,
                args.num_workers,
                args.same_line_y,
                args.max_gap,
                args.min_height_ratio,
                args.no_merge,
                args.absolute_thresholds,
                args.width_threshold,
            )
            total_recognized += n_rec
            total_files_done += 1
            log.info(f"Done: {jp} -> {out_path} ({n_rec} vùng text đã nhận dạng lại)")
        except Exception as e:
            log.exception(f"Lỗi khi xử lý {jp}: {e}")
        finally:
            # luôn giải phóng lock dù thành công, lỗi, hay bị Ctrl+C giữa chừng,
            # để tiến trình khác (nếu có) không bị treo chờ vô ích
            if not args.no_lock:
                release_lock(lock_path)

    log.info(
        f"Hoàn tất: {total_files_done}/{len(json_files)} file, "
        f"{total_skipped_locked} file bị skip vì đang khoá bởi tiến trình khác, "
        f"tổng {total_recognized} vùng text đã nhận dạng bằng VietOCR."
    )


if __name__ == "__main__":
    main()