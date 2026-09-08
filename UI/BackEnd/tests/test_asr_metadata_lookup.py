from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from search.asr_metadata import get_asr_for_keyframe_metadata


def test_reads_asr_6s_from_metadata_jsonl():
    result = get_asr_for_keyframe_metadata("L21_V001", keyframe_id="L21_V001_kf_0001")
    assert result is not None
    assert result["asr_text"]
    assert "Chào mừng quý vị" in result["asr_text"]
    assert result["asr_6s"] == result["asr_text"]
