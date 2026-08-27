#!/usr/bin/env python3
import os
import glob
import json
import time
from collections import Counter, defaultdict

ASR_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr"
KEYFRAMES_DIR = "/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes"

REQUIRED_TOP_FIELDS = ["keyframe_id"]
REQUIRED_ASR_SUBFIELDS = ["asr_text_6s", "has_speech"]

EMPTY_ASR_TOKENS = {"", None}

def read_keyframe_counts():
    kf_counts = {}
    for csv_path in glob.glob(os.path.join(KEYFRAMES_DIR, "*_keyframes.csv")):
        vid = os.path.basename(csv_path).replace("_keyframes.csv", "")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                kf_counts[vid] = max(0, len(lines) - 1)
        except Exception:
            pass
    if not kf_counts:
        for json_path in glob.glob(os.path.join(KEYFRAMES_DIR, "*_keyframes.json")):
            vid = os.path.basename(json_path).replace("_keyframes.json", "")
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    kf_counts[vid] = len(data) if isinstance(data, list) else 0
            except Exception:
                pass
    return kf_counts


def looks_like_garbage_text(text):
    if not isinstance(text, str):
        return True
    t = text.strip()
    if len(t) == 0:
        return False
    non_printable = sum(1 for c in t if ord(c) < 32 and c not in "\n\r\t")
    if non_printable > 0:
        return True
    if len(set(t)) <= 2 and len(t) >= 5:
        return True
    return False


def check_file(filepath, kf_counts):
    filename = os.path.basename(filepath)
    video_id = filename.replace("_asr.jsonl", "").replace("_asr.json", "")
    file_size = os.path.getsize(filepath)

    issues = []
    bad_records = []
    all_asr_texts = []
    non_empty_asr_texts = []
    valid_records = []
    total_lines = 0
    json_errors = 0

    expected_kf = kf_counts.get(video_id, None)

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()
    except Exception as e:
        issues.append(f"READ_ERROR: {e}")
        return {
            "filename": filename,
            "video_id": video_id,
            "file_size": file_size,
            "total_lines": 0,
            "valid_records": 0,
            "json_errors": 0,
            "expected_kf": expected_kf,
            "issues": issues,
            "bad_records": [],
            "sample_bad_content": raw_content[:500] if "raw_content" in dir() else "",
        }

    lines = raw_content.splitlines()
    total_lines = len(lines)

    if filepath.endswith(".jsonl"):
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError as je:
                json_errors += 1
                bad_records.append({
                    "line": line_num,
                    "error": f"JSON_DECODE: {je}",
                    "raw": stripped[:300],
                })
                continue
            valid_records.append((line_num, rec))
    else:
        try:
            data = json.loads(raw_content)
            items = data if isinstance(data, list) else [data]
            for idx, rec in enumerate(items, 1):
                valid_records.append((idx, rec))
        except json.JSONDecodeError as je:
            json_errors = total_lines
            issues.append(f"TOP_LEVEL_JSON_DECODE: {je}")
            return {
                "filename": filename,
                "video_id": video_id,
                "file_size": file_size,
                "total_lines": total_lines,
                "valid_records": 0,
                "json_errors": json_errors,
                "expected_kf": expected_kf,
                "issues": issues,
                "bad_records": [],
                "sample_bad_content": raw_content[:500],
            }

    kf_ids_seen = set()
    for line_num, rec in valid_records:
        rec_issues = []
        if not isinstance(rec, dict):
            bad_records.append({"line": line_num, "error": "NOT_DICT", "raw": str(rec)[:200]})
            continue
        for field in REQUIRED_TOP_FIELDS:
            if field not in rec:
                rec_issues.append(f"MISSING_{field.upper()}")
        kf_id = rec.get("keyframe_id")
        if kf_id:
            if kf_id in kf_ids_seen:
                rec_issues.append("DUPLICATE_KEYFRAME_ID")
            kf_ids_seen.add(kf_id)
        asr_meta = rec.get("asr_metadata")
        if asr_meta is None:
            rec_issues.append("MISSING_ASR_METADATA")
            asr_text = ""
        else:
            if not isinstance(asr_meta, dict):
                rec_issues.append("ASR_METADATA_NOT_DICT")
                asr_text = ""
            else:
                for sf in REQUIRED_ASR_SUBFIELDS:
                    if sf not in asr_meta:
                        rec_issues.append(f"MISSING_ASR_{sf.upper()}")
                asr_text = asr_meta.get("asr_text_6s", "")
        if isinstance(asr_text, str):
            all_asr_texts.append(asr_text)
            if asr_text.strip():
                non_empty_asr_texts.append(asr_text)
        if looks_like_garbage_text(asr_text):
            rec_issues.append("GARBAGE_ASR_TEXT")
        if rec_issues:
            bad_records.append({
                "line": line_num,
                "keyframe_id": kf_id,
                "video_id": rec.get("video_id"),
                "frame_idx": rec.get("frame_idx"),
                "pts_time": rec.get("pts_time"),
                "asr_text_preview": (asr_text[:150] if isinstance(asr_text, str) else str(asr_text)[:150]),
                "has_speech": asr_meta.get("has_speech") if isinstance(asr_meta, dict) else None,
                "issues": rec_issues,
            })

    num_valid = len(valid_records)
    unique_texts = set(all_asr_texts)
    unique_non_empty = set(non_empty_asr_texts)

    if num_valid == 0:
        issues.append("ZERO_VALID_RECORDS")
    else:
        if expected_kf is not None:
            ratio = num_valid / expected_kf if expected_kf > 0 else 0
            if ratio < 0.5:
                issues.append(f"TOO_FEW_RECORDS: {num_valid}/{expected_kf} ({ratio:.0%})")
            elif num_valid > expected_kf * 1.5 + 100:
                issues.append(f"TOO_MANY_RECORDS: {num_valid}/{expected_kf}")
        if json_errors > 0:
            issues.append(f"HAS_JSON_ERRORS: {json_errors} line(s)")
        if len(all_asr_texts) == len(non_empty_asr_texts) == 0:
            issues.append("ALL_ASR_MISSING")
        elif len(unique_non_empty) == 1 and len(non_empty_asr_texts) >= 10:
            repeated_text = next(iter(unique_non_empty))
            issues.append(f"ALL_ASR_SAME_TEXT: {len(non_empty_asr_texts)} records share 1 unique text ({len(repeated_text)} chars)")
        elif len(unique_texts) == 1 and "" in unique_texts and len(all_asr_texts) >= 10:
            issues.append("ALL_ASR_EMPTY_TEXT")
        if len(bad_records) > num_valid * 0.5 and num_valid > 0:
            issues.append(f"MAJORITY_BAD_RECORDS: {len(bad_records)}/{num_valid}")
        if file_size < 200 and num_valid < 5:
            issues.append("SUSPICIOUSLY_SMALL_FILE")

    sample_bad = raw_content[:500] if issues and total_lines < 10 else ""

    return {
        "filename": filename,
        "video_id": video_id,
        "file_size": file_size,
        "total_lines": total_lines,
        "valid_records": num_valid,
        "json_errors": json_errors,
        "expected_kf": expected_kf,
        "unique_asr_texts": len(unique_texts),
        "unique_non_empty_asr": len(unique_non_empty),
        "issues": issues,
        "bad_records": bad_records,
        "sample_bad_content": sample_bad,
        "_first_asr_texts_sample": list(unique_non_empty)[:3] if len(unique_non_empty) <= 5 else [],
    }


def main():
    t0 = time.time()
    kf_counts = read_keyframe_counts()
    print(f"[INFO] Loaded keyframe counts for {len(kf_counts)} videos.")

    pattern_jsonl = os.path.join(ASR_DIR, "*.jsonl")
    pattern_json = os.path.join(ASR_DIR, "*.json")
    files = sorted(glob.glob(pattern_jsonl) + glob.glob(pattern_json))
    print(f"[INFO] Scanning {len(files)} ASR metadata files in: {ASR_DIR}\n")

    corrupt_files = []
    clean_count = 0

    for i, fp in enumerate(files, 1):
        if i % 100 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"[PROGRESS] {i}/{len(files)} files scanned, elapsed={elapsed:.1f}s...")
        result = check_file(fp, kf_counts)
        if result["issues"] or result["bad_records"]:
            corrupt_files.append(result)
        else:
            clean_count += 1

    elapsed_total = time.time() - t0
    print(f"\n{'='*100}")
    print(f"[SCAN COMPLETE] {len(files)} total files | {clean_count} CLEAN | {len(corrupt_files)} CORRUPT/SUSPECT | elapsed={elapsed_total:.1f}s")
    print(f"{'='*100}\n")

    corrupt_files.sort(key=lambda r: (
        0 if "ZERO_VALID_RECORDS" in r["issues"] else
        0 if "ALL_ASR_SAME_TEXT" in r["issues"] else
        0 if "ALL_ASR_EMPTY_TEXT" in r["issues"] else
        0 if "TOP_LEVEL_JSON_DECODE" in r["issues"] else 1,
        -(len(r["bad_records"])),
    ))

    for idx, r in enumerate(corrupt_files, 1):
        print(f"\n{'#'*90}")
        print(f"# CORRUPT FILE #{idx}:  {r['filename']}")
        print(f"# Video ID: {r['video_id']}")
        print(f"# File size: {r['file_size']} bytes   Lines: {r['total_lines']}   Valid JSON records: {r['valid_records']}   JSON decode errors: {r['json_errors']}")
        expected_str = str(r['expected_kf']) if r['expected_kf'] is not None else "N/A"
        print(f"# Expected keyframes (from csv): {expected_str}   Unique asr texts: {r.get('unique_asr_texts','?')}   Unique non-empty: {r.get('unique_non_empty_asr','?')}")
        print(f"# ISSUES: {', '.join(r['issues']) if r['issues'] else '(issues from bad records only)'}")
        print(f"{'#'*90}")

        if r["sample_bad_content"]:
            print("  [RAW CONTENT SAMPLE (first 500 bytes)]:")
            print("    " + r["sample_bad_content"].replace("\n", "\n    "))
            print()

        if r["bad_records"]:
            preview_count = min(len(r["bad_records"]), 15)
            print(f"  [BAD RECORDS]: {len(r['bad_records'])} total, showing first {preview_count}:")
            for br in r["bad_records"][:preview_count]:
                print(f"    - line {br.get('line','?')}  kf={br.get('keyframe_id','?')}  frame_idx={br.get('frame_idx','?')}  pts={br.get('pts_time','?')}  has_speech={br.get('has_speech','?')}")
                print(f"      issues: {', '.join(br['issues'])}")
                if "raw" in br:
                    print(f"      raw: {br['raw']}")
                if br.get("asr_text_preview") is not None:
                    print(f"      asr_text_6s: {br['asr_text_preview']!r}")
            if len(r["bad_records"]) > preview_count:
                print(f"    ... ({len(r['bad_records']) - preview_count} more bad records omitted)")

    print(f"\n{'='*100}")
    print("[SUMMARY TABLE]")
    print(f"{'#':<4} {'FILE':<40} {'SIZE_B':<10} {'LINES':<8} {'VALID':<8} {'EXP_KF':<8} {'ISSUES'}")
    print("-" * 120)
    for idx, r in enumerate(corrupt_files, 1):
        exp_str = str(r['expected_kf']) if r['expected_kf'] is not None else "-"
        issues_str = ", ".join(r["issues"]) if r["issues"] else f"{len(r['bad_records'])} bad recs"
        print(f"{idx:<4} {r['filename']:<40} {r['file_size']:<10} {r['total_lines']:<8} {r['valid_records']:<8} {exp_str:<8} {issues_str[:60]}")

    print(f"\nTotal corrupt/suspect files: {len(corrupt_files)} / {len(files)}")
    counter = Counter()
    for r in corrupt_files:
        for iss in r["issues"]:
            iss_key = iss.split(":")[0] if ":" in iss else iss
            counter[iss_key] += 1
    if counter:
        print("Issue breakdown:")
        for k, v in counter.most_common():
            print(f"  {k:<35} {v}")


if __name__ == "__main__":
    main()
