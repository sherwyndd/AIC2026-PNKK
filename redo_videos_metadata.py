#!/usr/bin/env python3
"""Redo videos_metadata.csv cho toàn bộ / 1 list video.

Ghi đè lên file videos_metadata.csv hiện tại, dùng EXACT 11 cột y chang:
  video_id,file_name,fps,total_frames,duration,width,height,bitrate_kbps,backend,format,fourcc

- Ưu tiên ffprobe (backend=FFMPEG) → 100% khớp format cũ (L25_V010..V011 có backend FFMPEG).
- Fallback cv2 nếu không có ffprobe hoặc probe lỗi (backend=OPENCV).
- Giống redo_all_asr_videos.py: backup csv cũ trước, hỗ trợ --video_list / --video_ids / --all / --dry_run.

Usage:
  python redo_videos_metadata.py --all
  python redo_videos_metadata.py --video_list redo_keyframes_114.txt
  python redo_videos_metadata.py --video_ids L23_V018 L24_V005
  python redo_videos_metadata.py --dry_run --all
"""
import argparse, csv, glob, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# MẶC ĐỊNH ĐƯỜNG DẪN (giống ASR / fps_verify.py: tìm khắp nơi cho đến khi có mp4)
# ---------------------------------------------------------------------------
METADATA_ROOT = Path("/AIClub_NAS/core_baotg/nhan/dataset/metadata")
VIDEOS_META_CSV = METADATA_ROOT / "videos" / "videos_metadata.csv"
KEYFRAMES_CSV_DIR = METADATA_ROOT / "keyframes"

VIDEO_SEARCH_DIRS: List[Path] = [
    Path("/mlcv1/Datasets/HCMAI25/full"),
    Path("/mlcv2025/Datasets/HCMAI25/batch2/video"),
    Path("/workspace/dataset/HCMAI25/full"),
    Path("/workspace/dataset/HCMAI25/batch1"),
    Path("/AIClub_NAS/Datasets/HCMAI25/full"),
    Path("/AIClub_NAS/core_baotg/nhan/dataset"),
]

CSV_COLUMNS = [
    "video_id", "file_name", "fps", "total_frames", "duration",
    "width", "height", "bitrate_kbps", "backend", "format", "fourcc",
]
BACKUP_DIR = METADATA_ROOT / "videos" / "backup_before_redo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def all_video_ids_from_keyframes() -> List[str]:
    if not KEYFRAMES_CSV_DIR.is_dir():
        return []
    ids: List[str] = []
    for f in sorted(KEYFRAMES_CSV_DIR.glob("*_keyframes.csv")):
        vid = f.name[: -len("_keyframes.csv")]
        if vid:
            ids.append(vid)
    return ids


def read_video_list_txt(path: Path) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def find_video_file(video_id: str, extra_input_dirs: List[Path]) -> Optional[Path]:
    """Tìm <video_id>.mp4 (hoặc .MP4) trong các thư mục tìm được, trả về path đầu tiên khớp."""
    candidates = [f"{video_id}.mp4", f"{video_id}.MP4", f"{video_id}.mkv", f"{video_id}.MKV"]
    checked = set()
    search_order: List[Path] = list(extra_input_dirs) + list(VIDEO_SEARCH_DIRS)
    for base in search_order:
        if not base:
            continue
        bp = Path(base)
        bp_key = str(bp.resolve()) if bp.exists() else str(bp)
        if bp_key in checked:
            continue
        checked.add(bp_key)
        if not bp.is_dir():
            continue
        for cname in candidates:
            p = bp / cname
            if p.is_file():
                return p
    return None


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


def probe_ffprobe(path: Path, timeout: int = 45) -> Optional[Dict]:
    """Probe bằng ffprobe → dict khớp 11 cột csv (chưa tính video_id/file_name)."""
    if not shutil.which("ffprobe"):
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,duration,width,height,codec_name,pix_fmt,codec_tag_string,codec_tag",
            "-show_entries", "format=duration,size,bit_rate,format_name",
            "-of", "json", str(path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        data = json.loads(out.decode("utf-8", "ignore"))
    except Exception:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    stream = streams[0] if streams else {}

    rfps = _parse_frac(stream.get("r_frame_rate") or "")
    afps = _parse_frac(stream.get("avg_frame_rate") or "")
    fps = rfps if (rfps and rfps > 0) else (afps or None)

    nb_raw = stream.get("nb_frames")
    total_frames: Optional[int] = None
    try:
        if nb_raw:
            total_frames = int(nb_raw)
    except Exception:
        total_frames = None

    dur_raw = None
    for src in (stream.get("duration"), fmt.get("duration")):
        try:
            if src is not None:
                dur_raw = float(src); break
        except Exception:
            continue

    # Nếu nb_frames == null và có fps + duration → ước lượng (đôi khi stream tag này bị null)
    if (total_frames is None or total_frames == 0) and fps and dur_raw:
        total_frames = int(round(fps * dur_raw))
    # Tương tự duration: nếu null nhưng có nb_frames + fps
    if (dur_raw is None or dur_raw <= 0) and total_frames and fps:
        dur_raw = total_frames / fps

    width = None
    height = None
    try:
        if stream.get("width") is not None: width = int(stream["width"])
        if stream.get("height") is not None: height = int(stream["height"])
    except Exception:
        pass

    bitrate_kbps: Optional[float] = None
    try:
        br_str = fmt.get("bit_rate")
        if br_str:
            bitrate_kbps = round(float(br_str) / 1000.0, 2)
    except Exception:
        bitrate_kbps = None

    # format: mã pixel / container? ví dụ cũ ghi "64" — đây thường là codec_tag 0x34363268 (= h264 fourcc thường hiển thị 64?).
    # Thực tế cột format trong sample = "64" (chuỗi số). Dùng codec_tag_string nếu có hoặc format_name rút gọn.
    fmt_str: str = ""
    fourcc_str: str = ""
    try:
        fcc = stream.get("codec_tag_string") or ""
        if fcc and fcc != "" and fcc != "0":
            fourcc_str = fcc
        # fallback: dùng codec_name (h264, hevc,...) cho fourcc nếu codec_tag_string trống
        if not fourcc_str:
            fourcc_str = (stream.get("codec_name") or "").lower()
        # format: trong sample ghi 64 → đây thường là giá trị đầu của codec_tag,
        # hoặc lấy độ dài bits mỗi pixel của pix_fmt không phải. Ta fallback: nếu fourcc có độ dài và sample cũ là số,
        # thử lấy codec_tag số, hex dạng rút gọn → cuối cùng dùng format_name rút gọn (ví dụ "matroska,webm" → "mkv")
        try:
            tag_int = int(stream.get("codec_tag") or 0)
            if tag_int:
                # sample cũ ghi '64' (một số nguyên ngắn) — đây có vẻ là format = pix_fmt số bits.
                # Ta set format = str của tag_int & 0xFF như heuristic gần giống; nếu không dùng format_name.
                fmt_str = str(tag_int & 0xFF)
        except Exception:
            pass
        if not fmt_str:
            fn = (fmt.get("format_name") or "").split(",")[0]
            fmt_str = fn
    except Exception:
        fmt_str = ""
        fourcc_str = ""

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration": dur_raw,
        "width": width,
        "height": height,
        "bitrate_kbps": bitrate_kbps,
        "backend": "FFMPEG",
        "format": fmt_str,
        "fourcc": fourcc_str,
    }


def probe_cv2(path: Path) -> Optional[Dict]:
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
        dur = None
        if fps and nb:
            dur = nb / fps
        # fourcc via cap.get
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = ""
        try:
            fourcc = (fourcc_int & 0xFF).to_bytes(1, "little").decode("latin1", "ignore") + \
                     ((fourcc_int >> 8) & 0xFF).to_bytes(1, "little").decode("latin1", "ignore") + \
                     ((fourcc_int >> 16) & 0xFF).to_bytes(1, "little").decode("latin1", "ignore") + \
                     ((fourcc_int >> 24) & 0xFF).to_bytes(1, "little").decode("latin1", "ignore")
        except Exception:
            fourcc = ""
        return {
            "fps": fps,
            "total_frames": nb,
            "duration": dur,
            "width": w,
            "height": h,
            "bitrate_kbps": None,  # cv2 không có trực tiếp
            "backend": "OPENCV",
            "format": str(fourcc_int & 0xFF) if fourcc_int else "",
            "fourcc": fourcc,
        }
    finally:
        cap.release()


def probe_one(video_id: str, extra_input_dirs: List[Path],
              prefer_ffprobe: bool = True) -> Tuple[Optional[Dict], Optional[Path], str]:
    """Trả về (row_dict, video_path, probe_method). row_dict gồm 9 field của csv (không có video_id/file_name)."""
    vpath = find_video_file(video_id, extra_input_dirs)
    if vpath is None:
        return None, None, "VIDEO_NOT_FOUND"

    if prefer_ffprobe:
        r = probe_ffprobe(vpath)
        if r is not None:
            return r, vpath, "ffprobe"
    r = probe_cv2(vpath)
    if r is not None:
        return r, vpath, "cv2"
    # try lại ffprobe nếu bị skip
    if not prefer_ffprobe:
        r = probe_ffprobe(vpath)
        if r is not None:
            return r, vpath, "ffprobe"
    return None, vpath, "PROBE_FAILED"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Redo videos_metadata.csv (ghi đè format cũ).")
    ap.add_argument("--output_csv", type=str, default=str(VIDEOS_META_CSV),
                    help=f"Path ghi đè CSV đích (mặc định: {VIDEOS_META_CSV})")
    ap.add_argument("--videos_metadata_dir", type=str, default=str(METADATA_ROOT / "videos"),
                    help="Thư mục videos metadata (dùng để tạo backup)")
    ap.add_argument("--input_dirs", nargs="*", default=[],
                    help="Thêm các thư mục tìm mp4 (nếu không nằm trong VIDEO_SEARCH_DIRS)")
    # Video list args
    ap.add_argument("--all", action="store_true",
                    help="Làm lại tất cả video_id có file _keyframes.csv trong metadata/keyframes (873 videos)")
    ap.add_argument("--video_ids", nargs="*", default=[],
                    help="Truyền video ID trực tiếp: --video_ids L23_V018 L24_V005")
    ap.add_argument("--video_list", type=str, default="",
                    help="File txt mỗi dòng 1 video_id (ví dụ redo_keyframes_114.txt)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Chỉ probe và in kế hoạch, KHÔNG ghi đè CSV")
    ap.add_argument("--skip_backup", action="store_true",
                    help="Không backup file CSV cũ trước khi ghi đè")
    ap.add_argument("--prefer_cv2", action="store_true",
                    help="Ưu tiên OpenCV trước (default: ưu tiên ffprobe để backend=FFMPEG giống file cũ)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Số thread/process probe song song (default 1, bảo toàn CPU/IO)")
    args = ap.parse_args()

    # 1. Build target video list
    pool: List[str] = []
    if args.all:
        pool = all_video_ids_from_keyframes()
    if args.video_ids:
        pool.extend(list(args.video_ids))
    if args.video_list:
        vp = Path(args.video_list)
        if not vp.is_file():
            print(f"[ERROR] video_list not found: {vp}", file=sys.stderr)
            sys.exit(2)
        pool.extend(read_video_list_txt(vp))
    if not pool:
        # Default: tất cả 873 video (như --all)
        pool = all_video_ids_from_keyframes()

    # Dedup + giữ thứ tự
    seen = set(); targets = []
    for v in pool:
        if v in seen: continue
        seen.add(v); targets.append(v)
    del pool

    if not targets:
        print("[ERROR] Empty video list (and --all found 0 keyframes csv). Nothing to do.")
        sys.exit(1)

    extra_input_dirs: List[Path] = [Path(p) for p in (args.input_dirs or [])]
    output_csv = Path(args.output_csv)
    videos_meta_dir = Path(args.videos_metadata_dir)

    print(f"[PLAN] videos_metadata redo: {len(targets)} video(s)")
    print(f"       output_csv   = {output_csv}")
    print(f"       search dirs  = {[str(p) for p in (extra_input_dirs + VIDEO_SEARCH_DIRS) if p]}")
    print(f"       prefer probe = {'cv2' if args.prefer_cv2 else 'ffprobe (backend FFMPEG giống mẫu cũ)'}")
    print(f"       dry_run      = {args.dry_run}")
    print()

    # 2. Probe từng video (có thể song song nếu args.concurrency >1, mặc định tuần tự)
    rows: List[Dict] = []
    status_counts: Dict[str, int] = {}
    missing_videos: List[str] = []
    probe_errors: List[str] = []

    def _do_one(vid: str) -> Tuple[str, Optional[Dict]]:
        r_d, vp, method = probe_one(vid, extra_input_dirs, prefer_ffprobe=(not args.prefer_cv2))
        if r_d is None:
            if method == "VIDEO_NOT_FOUND":
                return "VIDEO_NOT_FOUND", None
            return "PROBE_FAILED", None
        # Xây row 11 cột
        row: Dict = {"video_id": vid}
        row["file_name"] = vp.name if vp else f"{vid}.mp4"
        # Đảm bảo các field còn lại có key tương ứng CSV_COLUMNS
        for k in CSV_COLUMNS[2:]:
            row[k] = r_d.get(k, "")
        # format float -> str số thập phân 2 (giống mẫu cũ: 25.0, 2204.64, 867.0)
        def _fmt(v, decimals=2):
            if v is None or v == "":
                return ""
            try:
                fv = float(v)
                if fv == int(fv) and decimals == 2:
                    return f"{fv:.1f}"
                return f"{fv:.{decimals}f}"
            except Exception:
                return str(v)
        row["fps"] = _fmt(row.get("fps"), 1)
        row["total_frames"] = "" if (row.get("total_frames") in (None, "")) else int(row["total_frames"])
        row["duration"] = _fmt(row.get("duration"), 2)
        row["width"] = "" if (row.get("width") in (None, "")) else int(row["width"])
        row["height"] = "" if (row.get("height") in (None, "")) else int(row["height"])
        row["bitrate_kbps"] = _fmt(row.get("bitrate_kbps"), 1)
        return method, row

    if args.concurrency and args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        # Dùng ThreadPoolExecutor (IO-bound ffprobe subprocess)
        with ThreadPoolExecutor(max_workers=min(max(1, int(args.concurrency)), 64)) as ex:
            futs = {ex.submit(_do_one, v): v for v in targets}
            for i, fut in enumerate(as_completed(futs), 1):
                vid = futs[fut]
                try:
                    st, row = fut.result()
                except Exception as e:
                    st, row = "EXCEPTION", None
                    probe_errors.append(f"{vid}: EXCEPTION {e!r}")
                status_counts[st] = status_counts.get(st, 0) + 1
                if row is not None:
                    rows.append(row)
                elif st == "VIDEO_NOT_FOUND":
                    missing_videos.append(vid)
                elif st != "PROBE_OK":
                    probe_errors.append(f"{vid}: {st}")
                if i % 50 == 0 or i == len(targets):
                    print(f"  ... processed {i}/{len(targets)}  status summary: {dict(status_counts)}")
    else:
        for i, vid in enumerate(targets, 1):
            try:
                st, row = _do_one(vid)
            except Exception as e:
                st, row = "EXCEPTION", None
                probe_errors.append(f"{vid}: EXCEPTION {e!r}")
            status_counts[st] = status_counts.get(st, 0) + 1
            if row is not None:
                rows.append(row)
            elif st == "VIDEO_NOT_FOUND":
                missing_videos.append(vid)
            elif st != "PROBE_OK":
                probe_errors.append(f"{vid}: {st}")
            if i % 50 == 0 or i == len(targets):
                print(f"  ... processed {i}/{len(targets)}  status summary: {dict(status_counts)}")

    # Sắp xếp rows theo video_id (giống order file cũ nếu có)
    rows.sort(key=lambda r: r.get("video_id", ""))

    print()
    print(f"[DONE] Probe finished. rows_writable={len(rows)} status={dict(status_counts)}")
    if missing_videos:
        print(f"[WARN] VIDEO_NOT_FOUND ({len(missing_videos)}): {missing_videos[:10]}{' ...' if len(missing_videos) > 10 else ''}")
    if probe_errors:
        print(f"[WARN] PROBE_FAILED ({len(probe_errors)}): {probe_errors[:10]}{' ...' if len(probe_errors) > 10 else ''}")

    if args.dry_run:
        print(f"\n[DRY_RUN OK] Sẽ ghi {len(rows)} dòng vào {output_csv}. Csv header = {CSV_COLUMNS}")
        # In vài mẫu để khớp format cũ
        if rows:
            print("  Sample row (khớp 11 cột):")
            for r in rows[:2]:
                print("   ", ", ".join(str(r.get(c, "")) for c in CSV_COLUMNS))
        print("[DRY_RUN END] Không có file nào được thay đổi.")
        return

    # 3. Backup CSV cũ (nếu tồn tại và không skip)
    if output_csv.is_file() and not args.skip_backup:
        backup_dir = Path(args.videos_metadata_dir) / "backup_before_redo"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{output_csv.stem}.{ts}.csv.bak"
        shutil.copy2(output_csv, backup_path)
        print(f"[BACKUP] CSV cũ đã lưu lại: {backup_path}")

    # 4. Ghi CSV mới (OVERWRITE file cũ, EXACT header và format số khớp mẫu)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[WRITE OK] Đã ghi {len(rows)} dòng vào: {output_csv}")
    print(f"           ({output_csv.stat().st_size} bytes)")

    # 5. In vài dòng đầu để confirm format giống file cũ
    with open(output_csv, "r", encoding="utf-8") as f:
        lines = [next(f, "").rstrip("\n") for _ in range(4)]
    print(f"[VERIFY 4 dòng đầu]:\n   " + "\n   ".join(lines))

if __name__ == "__main__":
    main()
