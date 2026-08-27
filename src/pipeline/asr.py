import os

import torch
from faster_whisper import WhisperModel

from .config import get_layout, get_output_dir
from .io import (
    asr_metadata_path,
    ensure_output_dirs,
    list_videos,
    load_metadata_table,
    load_video_list,
    save_metadata,
    _read_metadata,
)
from src.utils.logger import parse_pipeline_log


def _log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def _extract_raw_asr(video_path: str, model: WhisperModel, language: str, beam_size: int) -> list[dict]:
    segments, _ = model.transcribe(video_path, language=language, beam_size=beam_size, condition_on_previous_text=False)
    return [
        {
            "segment_start": float(s.start),
            "segment_end": float(s.end),
            "text": s.text.strip(),
        }
        for s in segments
    ]


def _map_asr_to_keyframe_window(keyframes_data: list[dict], asr_segments: list[dict], window_sec: float):
    if not asr_segments:
        for row in keyframes_data:
            row["asr_metadata"] = {
                "window_start": max(0.0, float(row["pts_time"]) - window_sec),
                "window_end": float(row["pts_time"]) + window_sec,
                "asr_text_6s": "",
                "has_speech": False,
                "raw_segments": [],
            }
        return keyframes_data

    aligned = []
    for row in keyframes_data:
        t_kf = float(row["pts_time"])
        w_start = max(0.0, t_kf - window_sec)
        w_end = t_kf + window_sec
        matches = [
            seg
            for seg in asr_segments
            if float(seg["segment_end"]) > w_start and float(seg["segment_start"]) < w_end
        ]
        aligned.append({
            **row,
            "asr_metadata": {
                "window_start": w_start,
                "window_end": w_end,
                "asr_text_6s": " ".join([seg["text"] for seg in matches]).strip(),
                "has_speech": len(matches) > 0,
                "raw_segments": matches,
            },
        })
    return aligned


def run_asr_whisper(
    input_dir,
    output_dir=None,
    mode="demo",
    asr_format="json",
    data_format="json",
    language="vi",
    beam_size=5,
    window_sec=3.0,
    model_size="medium",
    layout=None,
    logger=None,
    resume=True,
    force=False,
    log_file=None,
    video_list_path=None,
    wait=True,
    wait_interval=5,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = str(get_output_dir(mode, output_dir))
    layout = layout or get_layout(mode)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_output_dirs(output_dir, layout)
    stage_done = parse_pipeline_log(log_file).get("asr_whisper", False) if resume else False
    if stage_done and not force and not video_list_path:
        _log(logger, f"Skipping asr_whisper stage: already completed in log {log_file}")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
        }
    video_list = load_video_list(video_list_path) if video_list_path else None
    videos = list_videos(input_dir, mode, video_list)
    pending_videos = list(videos)

    if not pending_videos:
        _log(logger, "All videos already processed in asr_whisper stage.")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
        }

    cuda_available = torch.cuda.is_available()
    compute_type = "float16" if cuda_available else "int8"
    model = WhisperModel(
        model_size,
        device="cuda" if cuda_available else "cpu",
        compute_type=compute_type,
        num_workers=1,
    )
    import time
    processed = 0

    while pending_videos:
        deferred = []

        for video_file in pending_videos:
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(input_dir, video_file)
            output_asr_path = asr_metadata_path(output_dir, video_id, asr_format, layout)
            if os.path.exists(output_asr_path) and not force:
                _log(logger, f"Skipping ASR for {video_id}: existing output found at {output_asr_path}")
                continue

            if not force:
                alt_found = None
                for fmt in ["json", "jsonl", "csv"]:
                    if fmt == asr_format:
                        continue
                    alt_path = asr_metadata_path(output_dir, video_id, fmt, layout)
                    if os.path.exists(alt_path):
                        alt_found = alt_path
                        break
                if alt_found:
                    _log(logger, f"Skipping ASR for {video_id}: existing output found in alternate format ({alt_found}). Converting to {output_asr_path}...")
                    alt_df = _read_metadata(alt_found)
                    if alt_df is not None and not alt_df.empty:
                        records = alt_df.fillna("").to_dict(orient="records")
                        save_metadata(records, output_asr_path, asr_format)
                        _log(logger, f"Saved converted ASR metadata for {video_id} to {output_asr_path}")
                    continue

            _log(logger, f"Processing ASR video {video_id}")

            # Skip immediately if source video doesn't exist
            if not os.path.exists(video_path):
                _log(logger, f"Warning: Source video not found {video_path}, skipping.")
                continue

            shots_dir = os.path.join(output_dir, layout["metadata_dir"], "shots")
            shots_csv = os.path.join(shots_dir, f"{video_id}_shots.csv")
            shots_json = os.path.join(shots_dir, f"{video_id}_shots.json")
            shots_df = load_metadata_table(shots_csv, shots_json)

            keyframes_dir = os.path.join(output_dir, layout["metadata_dir"], "keyframes")
            keyframes_csv = os.path.join(keyframes_dir, f"{video_id}_keyframes.csv")
            keyframes_json = os.path.join(keyframes_dir, f"{video_id}_keyframes.json")
            keyframes_df = load_metadata_table(keyframes_csv, keyframes_json)

            if shots_df is None or keyframes_df is None or keyframes_df.empty:
                if wait:
                    _log(logger, f"Keyframes not ready for {video_id}, deferring to next pass...")
                    deferred.append(video_file)
                else:
                    _log(logger, f"Warning: Missing shots or keyframes metadata for {video_id}, skipping.")
                continue

            raw_segments = _extract_raw_asr(video_path, model, language, beam_size)
            keyframes_data = keyframes_df.fillna("").to_dict(orient="records")
            aligned = _map_asr_to_keyframe_window(keyframes_data, raw_segments, window_sec)

            save_metadata(aligned, output_asr_path, asr_format)
            _log(logger, f"Saved ASR metadata for {video_id} to {output_asr_path}")
            processed += 1

        if not deferred:
            break

        _log(logger, f"{len(deferred)} videos deferred (keyframes not ready). Waiting {wait_interval}s before retry pass...")
        time.sleep(wait_interval)
        pending_videos = deferred

    _log(logger, f"[PIPELINE] stage=asr_whisper status=stage_completed")


    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "processed_videos": processed,
    }
