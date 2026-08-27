#!/usr/bin/env python3
import json
import os
from collections import defaultdict

ASR_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"
KF_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes"
OUT_TXT = "/AIClub_NAS/core_baotg/phong/AIC_2026/corrupt_keyframes_list.txt"

kf_counts = {}
for fname in os.listdir(KF_DIR):
    if fname.endswith("_keyframes.csv"):
        vid = fname.replace("_keyframes.csv", "")
        with open(os.path.join(KF_DIR, fname), "r") as f:
            kf_counts[vid] = max(0, sum(1 for _ in f) - 1)

text_cluster = defaultdict(list)
garbage_rows_all = []
single_text_rows_all = []
per_video_summary = []

for fname in sorted(os.listdir(ASR_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    vid = fname.replace("_asr.jsonl", "")
    fpath = os.path.join(ASR_DIR, fname)
    try:
        with open(fpath, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        continue
    recs = []
    for l in lines:
        try:
            recs.append(json.loads(l))
        except Exception:
            pass
    if not recs:
        continue

    asr_texts = []
    garbage_these = []
    single_these = []

    for r in recs:
        asr = r.get("asr_metadata", {}) or {}
        t = asr.get("asr_text_6s", "") if isinstance(asr, dict) else ""
        ts = t.strip() if isinstance(t, str) else ""
        asr_texts.append(ts)

        row_meta = {
            "video_id": r.get("video_id", vid),
            "keyframe_id": r.get("keyframe_id"),
            "shot_id": r.get("shot_id"),
            "frame_idx": r.get("frame_idx"),
            "pts_time": r.get("pts_time"),
            "image_path": r.get("image_path"),
            "has_speech": asr.get("has_speech") if isinstance(asr, dict) else None,
            "n_raw_seg": len(asr.get("raw_segments", [])) if isinstance(asr, dict) and isinstance(asr.get("raw_segments"), list) else 0,
            "asr_text_6s": ts,
            "asr_window_start": asr.get("window_start") if isinstance(asr, dict) else None,
            "asr_window_end": asr.get("window_end") if isinstance(asr, dict) else None,
        }

        if ts:
            non_print = sum(1 for c in ts if ord(c) < 32 and c not in "\n\r\t")
            bare = ts.replace(" ", "")
            if non_print > 0 or (len(set(bare)) <= 2 and len(bare) >= 3):
                garbage_these.append(row_meta)

    total = len(asr_texts)
    non_empty_texts = [t for t in asr_texts if t]
    unique_non_empty = set(non_empty_texts)

    is_loop = len(unique_non_empty) == 1 and len(non_empty_texts) >= 10
    loop_text = list(unique_non_empty)[0] if is_loop else None

    if is_loop:
        text_cluster[loop_text].append(vid)
        for r in recs:
            asr = r.get("asr_metadata", {}) or {}
            t = asr.get("asr_text_6s", "") if isinstance(asr, dict) else ""
            ts = t.strip() if isinstance(t, str) else ""
            if ts == loop_text:
                single_these.append({
                    "video_id": r.get("video_id", vid),
                    "keyframe_id": r.get("keyframe_id"),
                    "shot_id": r.get("shot_id"),
                    "frame_idx": r.get("frame_idx"),
                    "pts_time": r.get("pts_time"),
                    "image_path": r.get("image_path"),
                    "has_speech": asr.get("has_speech") if isinstance(asr, dict) else None,
                    "n_raw_seg": len(asr.get("raw_segments", [])) if isinstance(asr, dict) and isinstance(asr.get("raw_segments"), list) else 0,
                    "asr_text_6s": ts,
                })
        single_text_rows_all.extend(single_these)
        per_video_summary.append({
            "category": "GROUP1_SINGLE_TEXT_LOOP",
            "video_id": vid,
            "file": fname,
            "expected_kf_csv": kf_counts.get(vid, "?"),
            "total_records": total,
            "non_empty_count": len(non_empty_texts),
            "unique_non_empty_asr": len(unique_non_empty),
            "loop_text_len": len(loop_text),
            "loop_text": loop_text,
            "corrupt_kf_count": len(single_these),
        })

    if garbage_these:
        garbage_rows_all.extend(garbage_these)
        per_video_summary.append({
            "category": "GROUP2_GARBAGE_ASR",
            "video_id": vid,
            "file": fname,
            "expected_kf_csv": kf_counts.get(vid, "?"),
            "total_records": total,
            "corrupt_kf_count": len(garbage_these),
        })


# ======== Write out ========
sep_thick = "=" * 120
sep_thin = "-" * 120

with open(OUT_TXT, "w", encoding="utf-8") as out:
    out.write(sep_thick + "\n")
    out.write("DANH SÁCH KEYFRAME CÓ ASR LỖI (Nhóm 1: lặp 1 dòng text / Nhóm 2: text rác)\n")
    out.write(f"Thư mục ASR: {ASR_DIR}\n")
    out.write(f"Thư mục Keyframes CSV: {KF_DIR}\n")
    out.write(sep_thick + "\n\n")

    # ============ TỔNG HỢP SỐ LƯỢNG ============
    out.write(">>> [TỔNG HỢP SỐ LƯỢNG LỖI]\n")
    out.write(sep_thin + "\n")

    n_video_grp1 = sum(1 for s in per_video_summary if s["category"] == "GROUP1_SINGLE_TEXT_LOOP")
    n_video_grp2 = sum(1 for s in per_video_summary if s["category"] == "GROUP2_GARBAGE_ASR")
    total_kf_grp1 = len(single_text_rows_all)
    total_kf_grp2 = len(garbage_rows_all)

    out.write(f"Tổng video bị lỗi: {len(set(s['video_id'] for s in per_video_summary))}\n")
    out.write(f"  - GROUP 1 (lặp 1 dòng text): {n_video_grp1} videos, {total_kf_grp1} keyframes lỗi\n")
    out.write(f"  - GROUP 2 (asr text rác)    : {n_video_grp2} videos, {total_kf_grp2} keyframes lỗi\n")
    out.write(f"  - Tổng cộng keyframes lỗi   : {total_kf_grp1 + total_kf_grp2}\n\n")

    # Breakdown GROUP1 theo từng chuỗi text lặp
    out.write("GROUP 1: phân bố theo text bị lặp lại:\n")
    for txt, vids in sorted(text_cluster.items(), key=lambda x: -len(x[1])):
        cnt_kf = 0
        for row in single_text_rows_all:
            if row["asr_text_6s"] == txt:
                cnt_kf += 1
        out.write(f"  [{len(txt)} chars] {txt!r}\n")
        out.write(f"      -> {len(vids)} videos, {cnt_kf} keyframes lỗi (asr_text_6s = câu này)\n")
    out.write("\n")

    # Breakdown GROUP2
    out.write("GROUP 2: phân bố theo text rác:\n")
    garbage_text_count = defaultdict(int)
    for g in garbage_rows_all:
        garbage_text_count[g["asr_text_6s"]] += 1
    for txt, c in sorted(garbage_text_count.items(), key=lambda x: -x[1]):
        out.write(f"  [{len(txt)} chars] {txt!r}  -> {c} keyframes\n")
    out.write("\n")

    # ============ CHI TIẾT TỪNG VIDEO ============
    out.write(sep_thick + "\n")
    out.write(">>> [CHI TIẾT TỪNG VIDEO]\n")
    out.write(sep_thick + "\n\n")

    for s in sorted(per_video_summary, key=lambda x: (0 if x["category"].startswith("GROUP1") else 1, x["video_id"])):
        out.write(sep_thin + "\n")
        out.write(f"[{s['category']}] Video: {s['video_id']}   File: {s['file']}\n")
        out.write(f"    KF (CSV expected): {s['expected_kf_csv']}   Actual records: {s['total_records']}   Số kf ASR lỗi: {s['corrupt_kf_count']}\n")
        if s["category"] == "GROUP1_SINGLE_TEXT_LOOP":
            out.write(f"    Non-empty ASR records: {s['non_empty_count']}   Unique non-empty ASR: {s['unique_non_empty_asr']}   Loop text ({s['loop_text_len']} chars): {s['loop_text']!r}\n")
        out.write(sep_thin + "\n")

        if s["category"] == "GROUP1_SINGLE_TEXT_LOOP":
            out.write(f"  Keyframes bị gán lặp cùng 1 dòng ASR (loop text):\n")
            cnt = 0
            for kf in single_text_rows_all:
                if kf["video_id"] != s["video_id"]:
                    continue
                pts = kf.get("pts_time")
                pts_s = f"{pts:.3f}s" if isinstance(pts, (int, float)) else str(pts)
                out.write(
                    f"    kf_id={kf.get('keyframe_id'):<22} "
                    f"frame={str(kf.get('frame_idx')):<8} "
                    f"pts={pts_s:<12} "
                    f"shot={str(kf.get('shot_id')):<26} "
                    f"has_sp={str(kf.get('has_speech')):<6} "
                    f"n_seg={kf.get('n_raw_seg')}   "
                    f"img={kf.get('image_path')}\n"
                )
                cnt += 1
            out.write(f"  [{cnt} rows trong video này]\n")
        else:
            out.write(f"  Keyframes có ASR text rác:\n")
            cnt = 0
            for kf in garbage_rows_all:
                if kf["video_id"] != s["video_id"]:
                    continue
                pts = kf.get("pts_time")
                pts_s = f"{pts:.3f}s" if isinstance(pts, (int, float)) else str(pts)
                out.write(
                    f"    kf_id={kf.get('keyframe_id'):<22} "
                    f"frame={str(kf.get('frame_idx')):<8} "
                    f"pts={pts_s:<12} "
                    f"shot={str(kf.get('shot_id')):<26} "
                    f"has_sp={str(kf.get('has_speech')):<6} "
                    f"n_seg={kf.get('n_raw_seg')}   "
                    f"win=[{kf.get('asr_window_start')},{kf.get('asr_window_end')}]   "
                    f"asr_text={kf.get('asr_text_6s')!r}   "
                    f"img={kf.get('image_path')}\n"
                )
                cnt += 1
            out.write(f"  [{cnt} rows trong video này]\n")
        out.write("\n")

    # ============ DANH SÁCH PHẲNG TẤT CẢ KF LỖI (để copy/lọc) ============
    out.write("\n" + sep_thick + "\n")
    out.write(">>> [DANH SÁCH PHẲNG - TẤT CẢ KEYFRAME LỖI]\n")
    out.write(sep_thick + "\n")
    out.write("CATEGORY\tVIDEO_ID\tKEYFRAME_ID\tFRAME_IDX\tPTS_TIME\tSHOT_ID\tHAS_SPEECH\tN_RAW_SEG\tASR_TEXT_LEN\tIMAGE_PATH\tASR_TEXT\n")
    all_rows = (
        [("GROUP1", r) for r in single_text_rows_all] +
        [("GROUP2", r) for r in garbage_rows_all]
    )
    all_rows.sort(key=lambda x: (x[1]["video_id"], x[1]["frame_idx"] if isinstance(x[1]["frame_idx"], int) else -1))
    for cat, r in all_rows:
        pts = r.get("pts_time")
        pts_s = f"{pts:.3f}" if isinstance(pts, (int, float)) else ""
        fi = r.get("frame_idx")
        fi_s = str(fi) if isinstance(fi, int) else ""
        out.write(
            f"{cat}\t"
            f"{r.get('video_id','')}\t"
            f"{r.get('keyframe_id','')}\t"
            f"{fi_s}\t"
            f"{pts_s}\t"
            f"{r.get('shot_id','')}\t"
            f"{r.get('has_speech','')}\t"
            f"{r.get('n_raw_seg',0)}\t"
            f"{len(r.get('asr_text_6s',''))}\t"
            f"{r.get('image_path','')}\t"
            f"{r.get('asr_text_6s','')}\n"
        )

print(f"[DONE] Đã xuất report ra: {OUT_TXT}")
print(f"  GROUP1 videos: {n_video_grp1}, kf lỗi: {total_kf_grp1}")
print(f"  GROUP2 videos: {n_video_grp2}, kf lỗi: {total_kf_grp2}")
print(f"  Tổng kf lỗi: {total_kf_grp1 + total_kf_grp2}")
print(f"  File size (bytes): {os.path.getsize(OUT_TXT)}")
