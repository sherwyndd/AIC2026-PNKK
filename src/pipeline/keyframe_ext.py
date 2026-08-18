import os

import cv2
import torch
from tqdm import tqdm

from src.utils.gpu_utils import free_vram
from src.utils.logger import parse_pipeline_log

from .clip_embedder import CLIPEmbedder, collect_shot_candidates, filter_candidates_by_rel_diff
from .config import get_layout, get_output_dir
from .io import (
    ensure_output_dirs,
    keyframe_image_relpath,
    keyframes_video_dir,
    list_videos,
    load_metadata_table,
    load_video_list,
    metadata_paths,
    save_metadata,
)
from .transnet import get_video_metadata


def _log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def extract_keyframes_for_video(
    video_path,
    video_id,
    output_dir,
    shots_data,
    clip_embedder,
    fps,
    layout,
    sample_step,
    rel_diff_threshold,
    logger=None,
):
    keyframes_data = []
    if not shots_data:
        return keyframes_data

    os.makedirs(keyframes_video_dir(output_dir, video_id, layout), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return keyframes_data

    for shot in shots_data:
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])
        shot_id = shot["shot_id"]

        candidates = collect_shot_candidates(cap, start_frame, end_frame, sample_step)
        if not candidates:
            continue

        frames_bgr = [frame for _, frame in candidates]
        embeddings = clip_embedder.encode_images(frames_bgr)
        selected = filter_candidates_by_rel_diff(candidates, embeddings, rel_diff_threshold)

        for actual_frame_idx, frame in selected:
            kf_id = f"{video_id}_kf_{len(keyframes_data) + 1:04d}"
            rel_img_path = keyframe_image_relpath(video_id, kf_id, layout)
            abs_img_path = os.path.join(output_dir, rel_img_path)
            try:
                saved = cv2.imwrite(abs_img_path, frame)
                if not saved:
                    _log(logger, f"Warning: cv2.imwrite failed for {abs_img_path}")
                    continue
            except Exception as exc:
                _log(logger, f"Error writing keyframe {abs_img_path}: {exc}")
                continue

            keyframes_data.append({
                "keyframe_id": kf_id,
                "shot_id": shot_id,
                "video_id": video_id,
                "frame_idx": actual_frame_idx,
                "pts_time": actual_frame_idx / fps if fps > 0 else 0,
                "image_path": rel_img_path,
            })

    cap.release()
    return keyframes_data


def run_keyframe_ext(
    input_dir,
    output_dir=None,
    mode="demo",
    out_format="csv",
    sample_step=8,
    rel_diff_threshold=0.4,
    clip_batch_size=8,
    clip_model_id="openai/clip-vit-large-patch14",
    shots_dir=None,
    video_list_path=None,
    layout=None,
    logger=None,
    resume=True,
    force=False,
    log_file=None,
    wait=True,
    wait_interval=5,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = str(get_output_dir(mode, output_dir))
    layout = layout or get_layout(mode)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_output_dirs(output_dir, layout)
    stage_done = parse_pipeline_log(log_file).get("keyframe_ext", False) if resume else False
    if stage_done and not force and not video_list_path:
        _log(logger, f"Skipping keyframe_ext stage: already completed in log {log_file}")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
            "shots_dir": shots_dir,
        }

    metadata_dir = layout["metadata_dir"]
    if shots_dir is None:
        shots_dir = os.path.join(output_dir, metadata_dir, "shots")
    else:
        shots_dir = os.path.abspath(shots_dir)

    os.makedirs(shots_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_embedder = CLIPEmbedder(device, model_id=clip_model_id, batch_size=clip_batch_size)

    video_list = load_video_list(video_list_path) if video_list_path else None
    videos = list_videos(input_dir, mode, video_list)
    if video_list and len(videos) < len(video_list):
        missing = [name for name in video_list if name not in set(os.listdir(input_dir))]
        if missing:
            _log(logger, f"Warning: listed videos not found in input_dir: {missing}")
    if video_list:
        _log(logger, f"Keyframe extraction video list provided ({len(video_list)} items); processing {len(videos)} existing videos")

    import time

    processed = 0
    pending_videos = list(videos)

    if not pending_videos:
        _log(logger, "No videos found for keyframe_ext stage.")
    else:
        while pending_videos:
            deferred = []  # videos that don't have shots yet — revisit next pass

            for video_file in tqdm(pending_videos, desc="Extracting keyframes"):
                video_id = os.path.splitext(video_file)[0]
                video_path = os.path.join(input_dir, video_file)
                paths = metadata_paths(output_dir, video_id, out_format, layout)

                if os.path.exists(paths["keyframes"]) and not force:
                    _log(logger, f"Skipping keyframe_ext video {video_id}: existing keyframes metadata found")
                    processed += 1
                    continue

                _log(logger, f"Starting keyframe extraction for video {video_id}")
                try:
                    # Skip immediately if source video doesn't exist (shots will never be created)
                    if not os.path.exists(video_path):
                        _log(logger, f"Warning: Source video not found {video_path}, skipping.")
                        continue

                    shots_csv = os.path.join(shots_dir, f"{video_id}_shots.csv")
                    shots_json = os.path.join(shots_dir, f"{video_id}_shots.json")
                    shots_df = load_metadata_table(shots_csv, shots_json)

                    if shots_df is None or shots_df.empty:
                        if wait:
                            _log(logger, f"Shots not ready for {video_id}, deferring to next pass...")
                            deferred.append(video_file)
                        else:
                            _log(logger, f"Warning: No shots metadata for {video_id}, skipping.")
                        continue

                    metadata = get_video_metadata(video_path, video_id)
                    if not metadata:
                        _log(logger, f"Warning: Cannot open video {video_path}, skipping.")
                        continue

                    shots_data = shots_df.to_dict(orient="records")
                    keyframes_data = extract_keyframes_for_video(
                        video_path,
                        video_id,
                        output_dir,
                        shots_data,
                        clip_embedder,
                        metadata["fps"],
                        layout,
                        sample_step,
                        rel_diff_threshold,
                        logger=logger,
                    )

                    if keyframes_data:
                        save_metadata(keyframes_data, paths["keyframes"], out_format)
                        _log(logger, f"Saved keyframes metadata for video {video_id} to {paths['keyframes']}")
                        processed += 1
                    else:
                        _log(logger, f"Warning: No keyframes extracted for {video_id}, skipping save.")
                except Exception as exc:
                    _log(logger, f"Error processing keyframe_ext video {video_id}: {exc}. Will retry this video on next run.")
                    continue

            if not deferred:
                break  # nothing left to wait for

            _log(logger, f"{len(deferred)} videos deferred (shots not ready). Waiting {wait_interval}s before retry pass...")
            time.sleep(wait_interval)
            pending_videos = deferred

    _log(logger, f"[PIPELINE] stage=keyframe_ext status=stage_completed")

    free_vram()
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "processed_videos": processed,
        "shots_dir": shots_dir,
    }
