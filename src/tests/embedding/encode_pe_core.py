#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encode_pe_core.py
==================
Encode toan bo keyframe bang PE-Core-L (qua open_clip) va luu embedding
ra file .npy + danh sach keyframe_id tuong ung (de tu build FAISS/index sau).

Cai dat truoc khi chay (Windows, khong can xformers/torchcodec):
    conda create -n aic2026_pecore python=3.11 -y
    conda activate aic2026_pecore
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install open_clip_torch pillow numpy tqdm

Cach chay:
    python encode_pe_core.py --images_dir D:\path\to\keyframes --output_dir D:\path\to\output

Cau truc thu muc anh mac dinh: tim tat ca file .jpg/.jpeg/.png de quy trong
--images_dir (ho tro ca cau truoc video_id/keyframe.jpg pho bien trong AIC).

Output:
    <output_dir>/embeddings.npy   -> float16 array (N, D), D=1024 voi PE-Core-L
    <output_dir>/keyframe_ids.json -> list N phan tu, keyframe_ids[i] khop voi embeddings[i]
    Ho tro RESUME: neu chay lai voi cung output_dir, script se bo qua anh da encode roi.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def find_all_images(images_dir):
    """Quet de quy toan bo anh trong images_dir, tra ve list (keyframe_id, full_path).
    keyframe_id = duong dan tuong doi (khong extension), dung dau '/' de nhat quan
    vi du: L21_V001/L21_V001_kf_0001
    """
    items = []
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                full_path = os.path.join(root, fn)
                rel_path = os.path.relpath(full_path, images_dir)
                kfid = os.path.splitext(rel_path)[0].replace(os.sep, "/")
                items.append((kfid, full_path))
    items.sort(key=lambda x: x[0])
    return items


class ImageListDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        kfid, path = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img)
            ok = True
        except Exception as e:
            print(f"[WARN] Loi doc anh {path}: {e}")
            tensor = torch.zeros(3, 336, 336)
            ok = False
        return tensor, kfid, ok


def collate_fn(batch):
    tensors = torch.stack([b[0] for b in batch])
    kfids = [b[1] for b in batch]
    oks = [b[2] for b in batch]
    return tensors, kfids, oks


def main():
    parser = argparse.ArgumentParser(description="Encode keyframes bang PE-Core-L (open_clip).")
    parser.add_argument("--images_dir", required=True, help="Thu muc chua toan bo keyframe (quet de quy).")
    parser.add_argument("--output_dir", required=True, help="Thu muc luu embeddings.npy + keyframe_ids.json")
    parser.add_argument("--model_id", default="hf-hub:timm/PE-Core-L-14-336")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_every", type=int, default=50, help="Luu checkpoint moi X batch (phong khi crash giua chung).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    ids_path = os.path.join(args.output_dir, "keyframe_ids.json")

    print(f"[INFO] Dang quet anh trong {args.images_dir} ...")
    all_items = find_all_images(args.images_dir)
    print(f"[INFO] Tim thay {len(all_items)} anh.")
    if not all_items:
        print("[ERROR] Khong tim thay anh nao. Kiem tra lai --images_dir.")
        return

    # RESUME: neu da co output truoc do, bo qua nhung id da encode
    done_ids = set()
    existing_embs = None
    existing_ids = []
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        print("[INFO] Phat hien output cu -> resume, bo qua anh da encode.")
        existing_embs = np.load(emb_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            existing_ids = json.load(f)
        done_ids = set(existing_ids)
        print(f"[INFO] Da co {len(done_ids)} embedding tu truoc.")

    remaining_items = [(kfid, path) for kfid, path in all_items if kfid not in done_ids]
    print(f"[INFO] Con lai {len(remaining_items)} anh can encode.")

    if not remaining_items:
        print("[INFO] Da encode xong toan bo, khong can chay them.")
        return

    print(f"[INFO] Dang load model {args.model_id} tren {args.device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_id)
    model = model.to(args.device).eval()
    if args.device == "cuda":
        model = model.half()

    dataset = ImageListDataset(remaining_items, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )

    all_new_embs = []
    all_new_ids = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (imgs, kfids, oks) in enumerate(tqdm(loader, desc="Encoding")):
            imgs = imgs.to(args.device)
            if args.device == "cuda":
                imgs = imgs.half()
                with torch.autocast("cuda", dtype=torch.float16):
                    feats = model.encode_image(imgs, normalize=True)
            else:
                feats = model.encode_image(imgs, normalize=True)

            feats = feats.float().cpu().numpy().astype(np.float16)
            all_new_embs.append(feats)
            all_new_ids.extend(kfids)

            # checkpoint dinh ky de tranh mat het neu crash giua chung
            if (batch_idx + 1) % args.save_every == 0:
                _save_checkpoint(emb_path, ids_path, existing_embs, existing_ids, all_new_embs, all_new_ids)

    _save_checkpoint(emb_path, ids_path, existing_embs, existing_ids, all_new_embs, all_new_ids)

    elapsed = time.time() - start_time
    n_new = len(all_new_ids)
    speed = n_new / elapsed if elapsed > 0 else 0
    print(f"\n[DONE] Encode xong {n_new} anh moi trong {elapsed:.1f}s ({speed:.1f} img/s).")
    print(f"[DONE] Tong so embedding hien co: {len(existing_ids) + n_new}")
    print(f"[DONE] Da luu tai: {emb_path} va {ids_path}")


def _save_checkpoint(
    emb_path,
    ids_path,
    existing_embs,
    existing_ids,
    new_embs_list,
    new_ids_list,
):
    if not new_embs_list:
        return

    new_embs = np.concatenate(new_embs_list, axis=0)

    if existing_embs is not None:
        merged_embs = np.concatenate([existing_embs, new_embs], axis=0)
        merged_ids = existing_ids + new_ids_list
    else:
        merged_embs = new_embs
        merged_ids = new_ids_list

    # Ghi file tạm rồi rename, tránh hỏng file nếu crash giữa lúc ghi
    tmp_emb = emb_path + ".tmp"
    tmp_ids = ids_path + ".tmp"

    with open(tmp_emb, "wb") as f:
        np.save(f, merged_embs)

    with open(tmp_ids, "w", encoding="utf-8") as f:
        json.dump(merged_ids, f, ensure_ascii=False)

    os.replace(tmp_emb, emb_path)
    os.replace(tmp_ids, ids_path)


if __name__ == "__main__":
    main()