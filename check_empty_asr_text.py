#!/usr/bin/env python3
"""Script check empty ASR text across all metadata jsonl files.

Scans all *_asr.jsonl files under:
    /AIClub_NAS/core_baotg/nhan/dataset/metadata/asr/

For each keyframe row, "empty text" is defined as any of:
  1. `asr_metadata.asr_text_6s` is None / "" / whitespace-only  (primary UI text)
  2. `asr_metadata.raw_segments` is missing or all entries have empty/whitespace `text`
  3. `asr_metadata.has_speech == False` together with no useful text
"""

import json
import os
import sys
import glob
import time
from collections import defaultdict

ASR_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"


def _wsp(s) -> bool:
    return s is None or (isinstance(s, str) and not s.strip())


def main() -> int:
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(ASR_DIR, "*_asr.jsonl")))
    if not files:
        print(f"[ERROR] No *_asr.jsonl files found under {ASR_DIR}", file=sys.stderr)
        return 1

    total_files = len(files)
    total_rows = 0
    rows_missing_asr_meta = 0
    rows_empty_text6s = 0
    rows_empty_raw_all = 0
    rows_nospeech_and_empty = 0
    rows_any_empty = 0

    per_group_empty = defaultdict(int)  # Lxx -> count
    per_video_empty = []  # list[(vid, total_kf, empty_kf)]
    per_file_row = defaultdict(int)  # vid -> total rows
    per_file_empty = defaultdict(int)  # vid -> empty rows

    sample_empty = []  # sample up to 15 empty rows

    for fpath in files:
        vid = os.path.basename(fpath).replace("_asr.jsonl", "")
        # group prefix like L21 / L22 / ...
        group = vid.split("_", 1)[0] if "_" in vid else vid[:3]

        with open(fpath, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                per_file_row[vid] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] JSON decode fail at {vid}:L{lineno}", file=sys.stderr)
                    continue

                am = row.get("asr_metadata") or {}
                if not isinstance(am, dict):
                    rows_missing_asr_meta += 1
                    per_file_empty[vid] += 1
                    per_group_empty[group] += 1
                    rows_any_empty += 1
                    if len(sample_empty) < 15:
                        sample_empty.append((vid, lineno, row.get("keyframe_id"), "MISSING_asr_metadata", ""))
                    continue

                text6s = am.get("asr_text_6s")
                raw_segs = am.get("raw_segments") or []
                has_speech = bool(am.get("has_speech", True))  # default True if missing

                t6s_empty = _wsp(text6s)
                raw_all_empty = True
                for seg in raw_segs:
                    if isinstance(seg, dict) and not _wsp(seg.get("text")):
                        raw_all_empty = False
                        break

                no_speech_empty = (not has_speech) and (t6s_empty or raw_all_empty)
                any_empty = t6s_empty or raw_all_empty

                if t6s_empty:
                    rows_empty_text6s += 1
                if raw_all_empty:
                    rows_empty_raw_all += 1
                if no_speech_empty:
                    rows_nospeech_and_empty += 1
                if any_empty:
                    rows_any_empty += 1
                    per_file_empty[vid] += 1
                    per_group_empty[group] += 1
                    if len(sample_empty) < 15:
                        sample_empty.append((
                            vid, lineno, row.get("keyframe_id"),
                            "T6S_EMPTY" if t6s_empty else ("RAW_EMPTY" if raw_all_empty else "NOSPEECH"),
                            (text6s or "")[:60],
                        ))

    # Summary stats
    t1 = time.time()
    pct = lambda n: (n / total_rows * 100.0) if total_rows else 0.0

    print("=" * 80)
    print("CHECK EMPTY ASR TEXT  —  SUMMARY REPORT")
    print("=" * 80)
    print(f"Directory         : {ASR_DIR}")
    print(f"Total JSONL files : {total_files}")
    print(f"Total keyframes   : {total_rows:,}")
    print(f"Elapsed time      : {t1 - t0:.2f}s")
    print("-" * 80)
    print(f"[A] Missing asr_metadata dict           : {rows_missing_asr_meta:>8,}   ({pct(rows_missing_asr_meta):.3f}%)")
    print(f"[B] asr_metadata.asr_text_6s EMPTY      : {rows_empty_text6s:>8,}   ({pct(rows_empty_text6s):.3f}%)")
    print(f"[C] ALL raw_segments[].text EMPTY       : {rows_empty_raw_all:>8,}   ({pct(rows_empty_raw_all):.3f}%)")
    print(f"[D] has_speech=False & NO text (either) : {rows_nospeech_and_empty:>8,}   ({pct(rows_nospeech_and_empty):.3f}%)")
    print("-" * 80)
    print(f"[X] ANY EMPTY (B or C or missing meta)  : {rows_any_empty:>8,}   ({pct(rows_any_empty):.3f}%)  ← tổng số keyframes có text bị trống")
    print(f"[OK] Có text hợp lệ                      : {total_rows - rows_any_empty:>8,}   ({100.0 - pct(rows_any_empty):.3f}%)")

    # Top 10 groups most empty
    if per_group_empty:
        print("\n" + "-" * 80)
        print("TOP GROUPS (Lxx) SỐ LƯỢNG KF TEXT RỖNG NHẤT:")
        print("-" * 80)
        top_groups = sorted(per_group_empty.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        g_total_empty = sum(per_group_empty.values())
        for g, c in top_groups:
            print(f"  {g:<6s} : {c:>6,} kf   ({c / max(rows_any_empty,1)*100.0:>5.2f}% / tất cả empty)")

    # Top 10 videos most empty
    print("\n" + "-" * 80)
    print("TOP 15 VIDEOS CÓ NHIỀU KF TEXT RỖNG NHẤT (kèm % / tổng kf video đó):")
    print("-" * 80)
    ranking = []
    for vid, total in per_file_row.items():
        e = per_file_empty.get(vid, 0)
        if e > 0:
            ranking.append((e, e / total * 100.0, vid, total))
    ranking.sort(reverse=True)
    print(f"  {'#':>2}  {'VIDEO_ID':<12s}  {'EMPTY':>7s}  {'TOTAL':>7s}  {'% KF RỖNG':>10s}")
    for i, (e, pctv, vid, tot) in enumerate(ranking[:15], 1):
        print(f"  {i:>2}  {vid:<12s}  {e:>7,}  {tot:>7,}  {pctv:>9.2f}%")

    # Videos 100% empty
    full_empty_vids = [(v, per_file_row[v]) for v, t, vid, tot in ranking if t >= 99.999]
    print(f"\n{'=' * 80}")
    print(f"VIDEOS CÓ 100% KF TEXT RỖNG (hoàn toàn):")
    print(f"{'=' * 80}")
    if full_empty_vids:
        for v, tot in full_empty_vids:
            print(f"  · {v:<14s}  {tot:>5,} kf  (toàn bộ text trống)")
    else:
        print("  (không có video nào 100% rỗng)")

    # Sample empty
    print("\n" + "=" * 80)
    print(f"MẪU {len(sample_empty)} DÒNG TEXT RỖNG (sample):")
    print("=" * 80)
    print(f"  {'VIDEO_ID':<12s}  {'LINE':>6s}  {'KEYFRAME_ID':<26s}  {'TYPE':<14s}  {'text6s (truncated)'}")
    for (vid, ln, kid, ttype, t6s) in sample_empty:
        t6s_disp = (t6s.replace("\n", " ") or "")[:50]
        kid_disp = (kid or "")[:26]
        print(f"  {vid:<12s}  {ln:>6,}  {kid_disp:<26s}  {ttype:<14s}  {t6s_disp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
