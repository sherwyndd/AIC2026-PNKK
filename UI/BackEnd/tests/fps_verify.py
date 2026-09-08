#!/usr/bin/env python3
"""FPS Verifier cho AIC 2026 dataset.

Cách dùng:
    python3 fps_verify.py <video_id> [video_id2 ...]
    python3 fps_verify.py --all L21           # tât cả video L21
    python3 fps_verify.py --sample 20         # lấy mẫu ngẫu nhiên 20 video trên toàn bộ dataset

Script làm 3 việc:
1) Đọc metadata CSV (<VID>_keyframes.csv) -> tính FPS từ frame_idx & pts_time
   (fps_median_delta + fps_polyfit trên toàn bộ chuỗi keyframes)
2) Nếu có ffprobe -> đọc FPS thật từ stream video .mp4
   (r_frame_rate, avg_frame_rate, duration, nb_frames)
3) So sánh -> báo khớp / lệch / cảnh báo VFR / lỗi CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Cấu hình đường dẫn (giống BackEnd/search/keyframe_manager.py)
# ---------------------------------------------------------------------------
METADATA_KEYFRAMES_DIR = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes")
KEYFRAME_ROOT = Path("/AIClub_NAS/core_baotg/nhan/dataset/keyframes")
VIDEO_SEARCH_DIRS = [
    "/AIClub_NAS/core_baotg/nhan/dataset/keyframes",
    "/AIClub_NAS/core_baotg/nhan/dataset/metadata/videos",
    "/AIClub_NAS/core_baotg/nhan/dataset",
    "/AIClub_NAS/Datasets/HCMAI25/full",
    "/AIClub_NAS/Datasets/Viettel_AI_Track_1",
    "/mlcv1/WorkingSpace/Personal/baotg/AIC_2026/dataset",
    "/workspace/dataset/HCMAI25/full",
    "/workspace/dataset/HCMAI25/batch1",
    "/workspace/dataset/HCMAI25/streaming",
    "/workspace/dataset/Viettel_AI_Track_1",
    "/workspace/dataset",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def all_video_ids() -> List[str]:
    if not METADATA_KEYFRAMES_DIR.is_dir():
        return []
    ids = []
    for f in METADATA_KEYFRAMES_DIR.glob("*_keyframes.csv"):
        m = re.match(r"^(.+)_keyframes\.csv$", f.name)
        if m:
            ids.append(m.group(1))
    return sorted(ids)


def find_video_path(video_id: str) -> Optional[Path]:
    for base in VIDEO_SEARCH_DIRS:
        p = Path(base) / f"{video_id}.mp4"
        if p.is_file():
            return p
        p = Path(base) / f"{video_id}.MP4"
        if p.is_file():
            return p
    return None


def load_keyframes_from_csv(video_id: str) -> List[Dict]:
    csv_path = METADATA_KEYFRAMES_DIR / f"{video_id}_keyframes.csv"
    if not csv_path.is_file():
        return []
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fi = int(row.get("frame_idx") or 0)
            except Exception:
                fi = 0
            try:
                pts = float(row.get("pts_time") or 0.0)
            except Exception:
                pts = 0.0
            out.append({
                "keyframe_id": (row.get("keyframe_id") or "").strip(),
                "frame_idx": fi,
                "pts_time": pts,
                "timestamp_ms": int(round(pts * 1000)),
            })
    out.sort(key=lambda x: (x["pts_time"], x["frame_idx"]))
    return out


def load_keyframes_from_folder(video_id: str) -> List[Dict]:
    d = KEYFRAME_ROOT / video_id
    if not d.is_dir():
        return []
    files = sorted(
        f.name for f in d.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    out = []
    for i, fname in enumerate(files):
        kf_id = os.path.splitext(fname)[0]
        m = re.search(r"kf_(\d+)", kf_id)
        # Fallback: hardcode 25fps để ước, đánh dấu estimated
        fps_est = 25.0
        sample_step = 7  # giả định mặc định: lấy 1 ảnh mỗi 7 frame
        fi = i * sample_step
        pts = fi / fps_est
        out.append({
            "keyframe_id": kf_id,
            "frame_idx": fi,
            "pts_time": pts,
            "timestamp_ms": int(round(pts * 1000)),
            "estimated": True,
        })
    return out


# ---------------------------------------------------------------------------
# Tính FPS từ metadata keyframes
# ---------------------------------------------------------------------------
def fps_from_keyframes(keyframes: List[Dict]) -> Dict:
    result = {
        "count": len(keyframes),
        "fps_median_delta": None,
        "fps_polyfit": None,
        "kf_frame_step_median": None,
        "kf_seconds_step_median": None,
        "time_span_s": None,
        "max_frame_offset_error": None,
        "outlier_count": 0,
        "is_estimated": any(kf.get("estimated") for kf in keyframes),
    }
    if len(keyframes) < 2:
        return result

    xs = [float(k["frame_idx"]) for k in keyframes]
    ys = [float(k["pts_time"]) for k in keyframes]
    result["time_span_s"] = ys[-1] - ys[0]

    # 1) FPS = 1 / median(Δpts/Δframe)
    deltas_fps = []
    frame_steps = []
    sec_steps = []
    for i in range(1, len(keyframes)):
        dfi = xs[i] - xs[i - 1]
        dpt = ys[i] - ys[i - 1]
        if dfi > 0 and dpt > 0:
            deltas_fps.append(dfi / dpt)
            frame_steps.append(dfi)
            sec_steps.append(dpt)
    if deltas_fps:
        result["fps_median_delta"] = float(statistics.median(deltas_fps))
        result["kf_frame_step_median"] = float(statistics.median(frame_steps))
        result["kf_seconds_step_median"] = float(statistics.median(sec_steps))

    # 2) Polyfit bậc 1: pts_time = a * frame + b -> fps = 1/a
    try:
        if len(xs) >= 2 and max(xs) > min(xs):
            a, b = _polyfit1(xs, ys)
            if a > 0:
                result["fps_polyfit"] = 1.0 / a
            # 3) Offset error (frames) so với đường hồi quy
            errs = []
            for xi, yi in zip(xs, ys):
                expected_frame = (yi - b) / a if a > 0 else xi
                errs.append(abs(xi - expected_frame))
            if errs:
                result["max_frame_offset_error"] = float(max(errs))
                # Outlier: lệch > 0.5 frame
                result["outlier_count"] = sum(1 for e in errs if e > 0.5)
    except Exception:
        pass
    return result


def _polyfit1(xs, ys):
    """Linear regression thủ công (không cần numpy)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    den = 0.0
    for xi, yi in zip(xs, ys):
        num += (xi - mx) * (yi - my)
        den += (xi - mx) ** 2
    if den == 0:
        return 0.0, my
    a = num / den
    b = my - a * mx
    return a, b


# ---------------------------------------------------------------------------
# ffprobe / OpenCV probe nguồn video
# ---------------------------------------------------------------------------
def _parse_frac(s: str) -> Optional[float]:
    if not s:
        return None
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            av, bv = float(a), float(b)
            if bv == 0:
                return None
            return av / bv
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def probe_video_ffprobe(path: Path) -> Optional[Dict]:
    if not shutil.which("ffprobe"):
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,width,height,codec_name,time_base",
            "-show_entries", "format=duration,size,bit_rate",
            "-of", "json", str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=30)
        data = json.loads(out.decode("utf-8", "ignore"))
    except Exception:
        return None

    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    r = _parse_frac(stream.get("r_frame_rate"))
    a = _parse_frac(stream.get("avg_frame_rate"))
    nb = stream.get("nb_frames")
    try:
        nb_i = int(nb) if nb else None
    except Exception:
        nb_i = None
    dur_s = None
    for src in (stream.get("duration"), fmt.get("duration")):
        try:
            if src is not None:
                dur_s = float(src)
                break
        except Exception:
            continue
    return {
        "r_frame_rate": r,
        "avg_frame_rate": a,
        "nb_frames": nb_i,
        "duration_s": dur_s,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "codec": stream.get("codec_name"),
        "time_base": stream.get("time_base"),
        "vfr": (r is not None and a is not None and abs(r - a) > 0.05),
    }


def probe_video_cv2(path: Path) -> Optional[Dict]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
        nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        dur_s = None
        if fps and nb:
            dur_s = nb / fps
        return {
            "r_frame_rate": fps,
            "avg_frame_rate": fps,
            "nb_frames": nb,
            "duration_s": dur_s,
            "width": w,
            "height": h,
            "codec": None,
            "time_base": None,
            "vfr": False,
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def fmt_fps(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:0.3f}"


def fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "N/A"
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def pick_meta_fps(meta: Dict) -> Optional[float]:
    return meta.get("fps_polyfit") or meta.get("fps_median_delta")


def classify(meta_fps: Optional[float], real_fps: Optional[float]) -> Tuple[str, str]:
    if meta_fps is None and real_fps is None:
        return "❌", "Không đủ dữ liệu"
    if meta_fps is None:
        return "ℹ️", f"Chỉ có nguồn video: {fmt_fps(real_fps)} FPS"
    if real_fps is None:
        return "ℹ️", f"Chỉ có metadata: {fmt_fps(meta_fps)} FPS (không probe được video)"
    rel = abs(meta_fps - real_fps) / max(real_fps, 1e-9) * 100
    if rel < 0.5:
        return "✅", f"KHỚP: metadata={fmt_fps(meta_fps)} vs video={fmt_fps(real_fps)} (hiệu {rel:0.2f}%)"
    if rel < 2.0:
        return "⚠️", f"LỆCH NHỎ: metadata={fmt_fps(meta_fps)} vs video={fmt_fps(real_fps)} (hiệu {rel:0.2f}%) — nên cập nhật videoFps state"
    return "❌", f"LỆCH LỚN: metadata={fmt_fps(meta_fps)} vs video={fmt_fps(real_fps)} (hiệu {rel:0.2f}%) — cần regenerate metadata?"


def verify_one(video_id: str, show_outliers: bool = False) -> Tuple[str, Optional[float], Optional[float]]:
    kfs = load_keyframes_from_csv(video_id)
    source_tag = "CSV metadata"
    if not kfs:
        kfs = load_keyframes_from_folder(video_id)
        source_tag = "Folder keyframes (ESTIMATED, step=7/25fps)"
    meta = fps_from_keyframes(kfs)
    meta_fps = pick_meta_fps(meta)

    vpath = find_video_path(video_id)
    real = probe_video_ffprobe(vpath) if vpath else None
    probe_method = "ffprobe"
    if real is None and vpath is not None:
        real = probe_video_cv2(vpath)
        probe_method = "cv2 (ffprobe không có)"
    real_fps = None
    if real:
        # Ưu tiên r_frame_rate (mặc định trình phát dùng)
        real_fps = real.get("r_frame_rate") or real.get("avg_frame_rate")

    # ---- In báo cáo ----
    W = 63
    print("═" * W)
    print(f" VIDEO: \033[1;96m{video_id}\033[0m   [nguồn kf: {source_tag}]")
    print("─" * W)
    print(" 📋 Metadata Keyframes:")
    print(f"   • Số keyframes         : {meta['count']}")
    if meta["kf_frame_step_median"] is not None:
        print(
            f"   • Keyframe step (med)  : ~ mỗi {meta['kf_frame_step_median']:0.1f} frame"
            f"  (≈ {meta['kf_seconds_step_median']:0.3f}s)"
        )
    print(
        f"   • FPS (Δpts trung vị)  : {fmt_fps(meta['fps_median_delta'])} FPS"
    )
    print(
        f"   • FPS (hồi quy pts~fi) : {fmt_fps(meta['fps_polyfit'])} FPS"
    )
    if meta["max_frame_offset_error"] is not None:
        print(
            f"   • Max offset (frames)  : {meta['max_frame_offset_error']:0.3f}"
            f"   | outliers >0.5 frame : {meta['outlier_count']}"
        )
    if meta["time_span_s"] is not None:
        print(f"   • Thời gian cover      : {meta['time_span_s']:0.3f}s (~{fmt_dur(meta['time_span_s'])})")

    print("─" * W)
    print(" 🎬 Nguồn video (%s):" % probe_method)
    if vpath is None:
        print("   • Không tìm thấy file MP4 trong các thư mục VIDEO_SEARCH_DIRS")
    else:
        print(f"   • Đường dẫn            : {vpath}")
        if real is None:
            print("   • Không probe được (cài ffmpeg để dùng ffprobe)")
        else:
            r = real.get("r_frame_rate")
            a = real.get("avg_frame_rate")
            nb = real.get("nb_frames")
            dur = real.get("duration_s")
            print(
                f"   • r_frame_rate         : "
                f"{r and f'{r:0.4f}'} FPS   "
                f"{'⚠️ VFR ' if real.get('vfr') else ''}"
            )
            print(f"   • avg_frame_rate       : {a and f'{a:0.4f}'} FPS")
            print(f"   • Tổng frames          : {nb or 'N/A'}")
            print(f"   • Duration             : {dur and f'{dur:0.3f}s ({fmt_dur(dur)})' or 'N/A'}")
            if dur and r:
                print(f"   • Dur×FPS tính fr      : ~{int(round(dur*r))}")
            if real.get("width") or real.get("codec"):
                print(
                    f"   • Resolution / Codec   : "
                    f"{real.get('width') or '?'}x{real.get('height') or '?'} , "
                    f"{real.get('codec') or '?'}"
                )

    status, msg = classify(meta_fps, real_fps)
    print("─" * W)
    print(f" {status} {msg}")
    print("═" * W)
    print()
    return status, meta_fps, real_fps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="Video IDs (L21_V001 L30_V064 …)")
    ap.add_argument("--all-batch", default=None, help="Duyệt toàn bộ batch ví dụ L21, L30…")
    ap.add_argument("--sample", type=int, default=0, help="Lấy mẫu N video ngẫu nhiên từ 873 video")
    ap.add_argument("--all", action="store_true", help="Duyệt toàn bộ 873 video (có thể lâu)")
    args = ap.parse_args(argv)

    target_ids: List[str] = []
    pool = all_video_ids()
    if args.all:
        target_ids = pool
    elif args.all_batch:
        prefix = args.all_batch.strip().rstrip("_") + "_"
        target_ids = [v for v in pool if v.startswith(prefix)]
    elif args.sample > 0:
        random.seed(0xC0FFEE)
        target_ids = random.sample(pool, min(args.sample, len(pool)))
    else:
        target_ids = list(args.videos)

    if not target_ids:
        print("❗ Chưa truyền video id. Ví dụ:\n   python3 fps_verify.py L30_V064 L21_V001\n   python3 fps_verify.py --sample 15\n   python3 fps_verify.py --all-batch L30\n")
        return 1

    print(f"\n🎞️  FPS Verifier · tổng cộng \033[1;92m{len(target_ids)}\033[0m video cần check.\n")

    # Tổng hợp
    fps_vals_meta = []
    fps_vals_real = []
    counts = {"✅": 0, "⚠️": 0, "❌": 0, "ℹ️": 0}
    issues = []
    for vid in target_ids:
        try:
            st, mf, rf = verify_one(vid)
        except Exception as e:
            print(f"❌ Lỗi xử lý {vid}: {e!r}\n")
            st, mf, rf = "❌", None, None
        k = st[0] if st and st[0] in counts else "ℹ️"
        counts[k] = counts.get(k, 0) + 1
        if mf is not None:
            fps_vals_meta.append(mf)
        if rf is not None:
            fps_vals_real.append(rf)
        if k in ("⚠️", "❌"):
            issues.append((st, vid, mf, rf))

    # ---- Tổng hợp cuối ----
    W = 63
    print("═" * W)
    print(" TỔNG KẾT  ({} video)".format(len(target_ids)))
    print("─" * W)
    print(f"   ✅ Khớp           : {counts.get('✅', 0)}")
    print(f"   ⚠️ Lệch nhỏ        : {counts.get('⚠️', 0)}")
    print(f"   ❌ Lệch lớn / Lỗi  : {counts.get('❌', 0)}")
    print(f"   ℹ️ Thiếu dữ liệu   : {counts.get('ℹ️', 0)}")
    if fps_vals_meta:
        lo, hi = min(fps_vals_meta), max(fps_vals_meta)
        print(f"\n 📋 Meta FPS     : min={lo:0.3f}  max={hi:0.3f}  median={statistics.median(fps_vals_meta):0.3f}")
        if abs(hi - lo) < 0.1:
            print("    ✨ → Metadata FPS TRÊN TOÀN BỘ KHỚP NHAU (cùng 1 chuẩn)")
        else:
            print("    ⚠️ → Metadata FPS KHÔNG ĐỒNG NHẤT giữa các video!")
    if fps_vals_real:
        lo, hi = min(fps_vals_real), max(fps_vals_real)
        print(f" 🎬 Video FPS    : min={lo:0.3f}  max={hi:0.3f}  median={statistics.median(fps_vals_real):0.3f}")
        if abs(hi - lo) < 0.1:
            print("    ✨ → Nguồn video FPS TRÊN TOÀN BỘ KHỚP NHAU (cùng 1 chuẩn)")
        else:
            print("    ⚠️ → Nguồn video FPS KHÔNG ĐỒNG NHẤT giữa các video!")
    if issues:
        print("\n DANH SÁCH CẦN XEM LẠI:")
        for st, vid, mf, rf in issues:
            print(f"   {st} {vid:16s}  meta={fmt_fps(mf)}  video={fmt_fps(rf)}")
    print("═" * W)
    return 0


if __name__ == "__main__":
    sys.exit(main())
