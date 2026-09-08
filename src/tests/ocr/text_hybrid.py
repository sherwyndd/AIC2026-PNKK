import json
import re
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    "/AIClub_NAS/core_baotg/nhan/data-AIC2025"
)

INPUT_DIR = PROJECT_ROOT / "outputs" / "ocr_vietocr"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ocr_hybrid"

TIME_PATTERN = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


# ============================================================
# HYBRID OCR
# ============================================================

def hybrid_ocr(text_paddle: str, text_vietocr: str) -> str:
    """
    Hybrid OCR:

    - Phần trước timestamp: lấy từ PaddleOCR
    - Timestamp: lấy từ PaddleOCR
    - Phần sau timestamp: lấy từ VietOCR

    Ví dụ:

    Paddle:
        HTV 9D 06:49:33 , Israel Hezbollah không kích d di ...

    VietOCR:
        HTM 06:49:33 , Israel Hezbollah không kích dữ dội ...

    Output:
        HTV 9D 06:49:33 , Israel Hezbollah không kích dữ dội ...
    """

    text_paddle = text_paddle or ""
    text_vietocr = text_vietocr or ""

    if not text_paddle:
        return text_vietocr

    if not text_vietocr:
        return text_paddle

    paddle_match = TIME_PATTERN.search(text_paddle)
    vietocr_match = TIME_PATTERN.search(text_vietocr)

    # Không tìm được timestamp ở một trong hai text
    # -> giữ nguyên VietOCR
    if paddle_match is None or vietocr_match is None:
        return text_vietocr

    # Phần đầu từ PaddleOCR
    paddle_prefix = text_paddle[:paddle_match.start()]

    # Timestamp từ PaddleOCR
    timestamp = text_paddle[
        paddle_match.start():paddle_match.end()
    ]

    # Phần sau timestamp từ VietOCR
    vietocr_suffix = text_vietocr[
        vietocr_match.end():
    ]

    return (
        paddle_prefix
        + timestamp
        + vietocr_suffix
    ).strip()


# ============================================================
# PROCESS ONE JSON
# ============================================================

def process_json(input_path: Path, output_path: Path):
    print(f"Processing: {input_path.name}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected JSON list, got {type(data).__name__}: "
            f"{input_path}"
        )

    total = 0
    hybrid_changed = 0
    no_timestamp = 0

    for item in data:

        text_paddle = item.get("text", "")
        text_vietocr = item.get("text_vietocr", "")

        paddle_match = TIME_PATTERN.search(text_paddle or "")
        vietocr_match = TIME_PATTERN.search(text_vietocr or "")

        if paddle_match is None or vietocr_match is None:
            no_timestamp += 1

        hybrid = hybrid_ocr(
            text_paddle,
            text_vietocr,
        )

        item["text_hybrid"] = hybrid

        total += 1

        if hybrid != text_vietocr:
            hybrid_changed += 1

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"  records       : {total}"
    )
    print(
        f"  hybrid changed: {hybrid_changed}"
    )
    print(
        f"  no timestamp  : {no_timestamp}"
    )
    print(
        f"  output        : {output_path}"
    )
    print()


# ============================================================
# PROCESS ALL JSON
# ============================================================

def main():

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_files = sorted(
        INPUT_DIR.glob("*.json")
    )

    if not json_files:
        print(
            f"No JSON files found in {INPUT_DIR}"
        )
        return

    print("=" * 70)
    print("HYBRID OCR")
    print("=" * 70)
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Files : {len(json_files)}")
    print("=" * 70)
    print()

    success = 0
    failed = 0

    for input_path in json_files:

        output_path = (
            OUTPUT_DIR / input_path.name
        )

        try:
            process_json(
                input_path,
                output_path,
            )
            success += 1

        except Exception as e:
            failed += 1

            print(
                f"ERROR: {input_path.name}"
            )
            print(
                f"       {type(e).__name__}: {e}"
            )
            print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total files : {len(json_files)}")
    print(f"Success     : {success}")
    print(f"Failed      : {failed}")
    print(f"Output      : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()