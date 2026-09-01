#!/usr/bin/env python3
import json
import os
from collections import defaultdict, Counter

ASR_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"
KF_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes"
OUT_TXT = "/AIClub_NAS/core_baotg/phong/AIC_2026/corrupt_keyframes_list.txt"
OUT_VIDEOS_TXT = "/AIClub_NAS/core_baotg/phong/AIC_2026/redo_videos_list.txt"

INTRO_PATTERNS_EXACT = [
    "Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn",
    "Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn",
    "Hãy subscribe cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn",
    "Hãy đăng ký kênh để ủng hộ kênh nhé các bạn!",
    "Các bạn hãy đăng kí cho kênh lalaschool Để không bỏ lỡ những video hấp dẫn.",
    "Hãy đăng ký kênh để ủng hộ kênh lalaschool Để không bỏ lỡ những video hấp dẫn",
    "Hãy đăng ký kênh để xem những video hấp dẫn.",
    "Đăng ký kênh để ủng hộ kênh nhé!",
]

INTRO_SUBSTR = [
    "Ghiền Mì Gõ Để không bỏ lỡ",
    "La La School Để không bỏ lỡ",
    "lalaschool Để không bỏ lỡ",
    "Đăng ký kênh để ủng hộ kênh",
    "subscribe cho kênh",
]

CORRUPT_RATIO_THRESHOLD = 0.50
HIGH_REPEAT_TEXT_THRESHOLD = 15

kf_counts = {}
for fname in os.listdir(KF_DIR):
    if fname.endswith("_keyframes.csv"):
        vid = fname.replace("_keyframes.csv", "")
        with open(os.path.join(KF_DIR, fname), "r") as f:
            kf_counts[vid] = max(0, sum(1 for _ in f) - 1)

per_video_result = []
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
    for r in recs:
        asr = r.get("asr_metadata", {}) or {}
        t = asr.get("asr_text_6s", "") if isinstance(asr, dict) else ""
        ts = t.strip() if isinstance(t, str) else ""
        asr_texts.append(ts)

    total = len(asr_texts)
    non_empty_idx = [i for i, t in enumerate(asr_texts) if t]
    n_non_empty = len(non_empty_idx)
    if n_non_empty == 0:
        continue

    unique_texts_with_non_empty = [asr_texts[i] for i in non_empty_idx]
    text_counts = Counter(unique_texts_with_non_empty)

    is_garbage_flags = {}
    for i in non_empty_idx:
        t = asr_texts[i]
        non_print = sum(1 for c in t if ord(c) < 32 and c not in "\n\r\t")
        bare = t.replace(" ", "")
        if non_print > 0 or (len(set(bare)) <= 2 and len(bare) >= 3):
            is_garbage_flags[i] = "GARBAGE"
            continue
        if t in INTRO_PATTERNS_EXACT:
            is_garbage_flags[i] = "INTRO_EXACT"
            continue
        if text_counts[t] >= HIGH_REPEAT_TEXT_THRESHOLD:
            is_garbage_flags[i] = "HIGH_REPEAT"
            continue
        for sub in INTRO_SUBSTR:
            if sub in t and text_counts[t] >= 8:
                is_garbage_flags[i] = "INTRO_SUBSTR"
                break

    n_corrupt = len(is_garbage_flags)
    ratio = n_corrupt / n_non_empty

    if ratio >= CORRUPT_RATIO_THRESHOLD:
        counts_by_reason = Counter(is_garbage_flags.values())
        corrupt_text_counter = Counter()
        corrupt_text_reasons = {}
        for i, reason in is_garbage_flags.items():
            t = asr_texts[i]
            corrupt_text_counter[t] += 1
            if t not in corrupt_text_reasons:
                corrupt_text_reasons[t] = reason
        top_corrupt_with_meta = [
            (txt, cnt, corrupt_text_reasons.get(txt, "?"))
            for txt, cnt in corrupt_text_counter.most_common(5)
        ]
        per_video_result.append({
            "video_id": vid,
            "file": fname,
            "expected_kf_csv": kf_counts.get(vid, "?"),
            "total_records": total,
            "non_empty": n_non_empty,
            "unique_non_empty": len(set(unique_texts_with_non_empty)),
            "n_corrupt": n_corrupt,
            "ratio": ratio,
            "reason_breakdown": dict(counts_by_reason),
            "top_corrupt_texts": top_corrupt_with_meta,
        })

sep_thick = "=" * 120
sep_thin = "-" * 120

with open(OUT_TXT, "w", encoding="utf-8") as out:
    out.write(sep_thick + "\n")
    out.write("DANH SÁCH VIDEO CÓ ASR NHIỀU TEXT RÁC CẦN REDO (video-level filter)\n")
    out.write(f"Thư mục ASR: {ASR_DIR}\n")
    out.write(f"Thư mục Keyframes CSV: {KF_DIR}\n")
    out.write(f"Threshold: (số text rác / số non-empty text) >= {CORRUPT_RATIO_THRESHOLD:.0%} thì list video cần redo\n")
    out.write(f"Định nghĩa TEXT RÁC per-record:\n")
    out.write(f"  * GARBAGE     : non-printable chars, hoặc chỉ <=2 ký tự unique lặp (vd '8 8 8 8', 'Ch Ch')\n")
    out.write(f"  * INTRO_EXACT : text khớp 1 trong 8 câu YouTube intro chuẩn\n")
    out.write(f"  * HIGH_REPEAT : text bất kỳ xuất hiện >= {HIGH_REPEAT_TEXT_THRESHOLD} lần trong cùng video (drift)\n")
    out.write(f"  * INTRO_SUBSTR: text chứa intro substring (vd 'Ghiền Mì Gõ Để không bỏ lỡ') VÀ xuất hiện >= 8 lần\n")
    out.write(sep_thick + "\n\n")

    out.write(">>> [TỔNG HỢP]\n")
    out.write(sep_thin + "\n")
    total_videos = len(per_video_result)
    total_corrupt_kf_estimate = sum(s["n_corrupt"] for s in per_video_result)
    out.write(f"Tổng số VIDEO cần redo ASR: {total_videos}\n")
    out.write(f"Tổng số keyframes bị đánh dấu text rác (ước tính): {total_corrupt_kf_estimate}\n\n")

    out.write("Breakdown theo loại text rác (tính trên toàn danh sách video):\n")
    global_reasons = Counter()
    for s in per_video_result:
        for r, c in s["reason_breakdown"].items():
            global_reasons[r] += c
    for r, c in global_reasons.most_common():
        out.write(f"  {r:<15} : {c} records\n")
    out.write("\n")

    out.write(">>> [DANH SÁCH VIDEO] (sắp xếp theo ratio giảm dần)\n")
    out.write(sep_thick + "\n")
    out.write(f"{'STT':<5}{'VIDEO_ID':<14}{'KF_CSV':<8}{'ASR_ROWS':<10}{'NON_EMPTY':<10}{'N_CORRUPT':<11}{'RATIO':<8}{'REASON_BREAKDOWN':<55}{'TOP_RAC_TEXT'}\n")
    out.write(sep_thin + "\n")
    for stt, s in enumerate(sorted(per_video_result, key=lambda x: -x["ratio"]), 1):
        rb_parts = [f"{r}={c}" for r, c in sorted(s["reason_breakdown"].items())]
        rb_str = ", ".join(rb_parts)
        top1 = s["top_corrupt_texts"][0][0][:70] if s["top_corrupt_texts"] else ""
        top1_cnt = s["top_corrupt_texts"][0][1] if s["top_corrupt_texts"] else 0
        out.write(
            f"{stt:<5}"
            f"{s['video_id']:<14}"
            f"{str(s['expected_kf_csv']):<8}"
            f"{s['total_records']:<10}"
            f"{s['non_empty']:<10}"
            f"{s['n_corrupt']:<11}"
            f"{s['ratio']*100:>5.1f}%  "
            f"{rb_str:<55}"
            f"[{top1_cnt}x] {top1!r}\n"
        )
    out.write("\n")

    out.write(">>> [CHI TIẾT TỪNG VIDEO]\n")
    out.write(sep_thick + "\n\n")
    for s in sorted(per_video_result, key=lambda x: x["video_id"]):
        out.write(sep_thin + "\n")
        out.write(f"VIDEO: {s['video_id']}   File: {s['file']}\n")
        out.write(f"  KF (CSV expected): {s['expected_kf_csv']}   ASR rows: {s['total_records']}   Non-empty: {s['non_empty']}   Unique non-empty: {s['unique_non_empty']}\n")
        out.write(f"  Text rác: {s['n_corrupt']}/{s['non_empty']} = {s['ratio']*100:.1f}%  (>= {CORRUPT_RATIO_THRESHOLD:.0%} → REDO)\n")
        rb_parts = [f"{r}={c}" for r, c in sorted(s["reason_breakdown"].items())]
        out.write(f"  Loại rác: {', '.join(rb_parts)}\n")
        out.write(f"  Top 5 text rác phổ biến nhất trong video này:\n")
        for txt, c, reason in s["top_corrupt_texts"]:
            out.write(f"    [{reason:<15}] [{c:<4}x] len={len(txt):<4}  {txt[:150]!r}\n")
        out.write("\n")

    out.write("\n" + sep_thick + "\n")
    out.write(">>> [DANH SÁCH PHẲNG VIDEO_ID CẦN REDO] (copy dòng dưới để pass vào script redo)\n")
    out.write(sep_thick + "\n")
    for s in sorted(per_video_result, key=lambda x: x["video_id"]):
        out.write(f"{s['video_id']}\n")
    out.write(f"\n# END. {total_videos} videos total\n")

print(f"[DONE] Đã xuất report ra: {OUT_TXT}")
print(f"  Tổng video cần redo: {total_videos} / {len([fn for fn in os.listdir(ASR_DIR) if fn.endswith('.jsonl')])} files jsonl")
print(f"  Text rác record ước tính: {total_corrupt_kf_estimate}")
print(f"  File size (bytes): {os.path.getsize(OUT_TXT)}")
print()
print(f"  Top 10 videos theo ratio rác:")
for stt, s in enumerate(sorted(per_video_result, key=lambda x: -x["ratio"])[:10], 1):
    print(f"    {stt:>2}. {s['video_id']:<14}  ratio={s['ratio']*100:>5.1f}%  ({s['n_corrupt']}/{s['non_empty']})  kiểu rác: {s['reason_breakdown']}")
