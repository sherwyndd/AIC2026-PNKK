import os

import torch
from tqdm import tqdm

from src.utils.gpu_utils import free_vram

from .config import get_layout, get_output_dir
from .io import ensure_output_dirs, list_videos, load_metadata_table, load_video_list, metadata_paths, save_metadata
from src.utils.logger import parse_pipeline_log
from .transnet import get_video_metadata, load_transnet_model, segment_video


def _log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def run_shot_cut(
    input_dir,
    output_dir=None,
    mode="demo",
    out_format="csv",
    video_list_path=None,
    layout=None,
    logger=None,
    resume=True,
    force=False,
    log_file=None,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = str(get_output_dir(mode, output_dir))
    layout = layout or get_layout(mode)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_output_dirs(output_dir, layout)
    stage_done = parse_pipeline_log(log_file).get("shot_cut", False) if resume else False
    if stage_done and not force and not video_list_path:
        _log(logger, f"Skipping shot_cut stage: already completed in log {log_file}")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
            "metadata_files": {},
        }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_transnet_model(device)
    if model is None:
        raise RuntimeError("TransNetV2 is unavailable. Check installation and weights.")

    video_list = load_video_list(video_list_path) if video_list_path else None
    videos = list_videos(input_dir, mode, video_list)
    if video_list and len(videos) < len(video_list):
        missing = [name for name in video_list if name not in set(os.listdir(input_dir))]
        if missing:
            _log(logger, f"Warning: listed videos not found in input_dir: {missing}")
    if video_list:
        _log(logger, f"Shot cut video list provided ({len(video_list)} items); processing {len(videos)} existing videos")

    videos_metadata_path = metadata_paths(output_dir, "placeholder", out_format, layout)["videos"]
    existing_videos_meta = load_metadata_table(
        videos_metadata_path,
        videos_metadata_path.replace(f".{out_format}", ".json"),
    )
    all_videos_metadata = []
    if existing_videos_meta is not None:
        all_videos_metadata = existing_videos_meta.to_dict(orient="records")

    pending_videos = list(videos)
    processed = 0
    if not pending_videos:
        _log(logger, "No videos found for shot_cut stage.")
    else:
        for video_file in tqdm(pending_videos, desc="Cutting shots"):
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(input_dir, video_file)
            paths = metadata_paths(output_dir, video_id, out_format, layout)

            if os.path.exists(paths["shots"]) and not force:
                _log(logger, f"Skipping shot_cut video {video_id}: existing shots metadata found")
                processed += 1
                continue

            _log(logger, f"Starting shot_cut for video {video_id}")
            try:
                metadata, shots_data = segment_video(video_path, video_id, model, device)
                if shots_data:
                    save_metadata(shots_data, paths["shots"], out_format)
                    _log(logger, f"Saved shots metadata for video {video_id} to {paths['shots']}")
                if metadata:
                    all_videos_metadata.append(metadata)
                    processed += 1
                    _log(logger, f"Completed shot_cut video {video_id}")
                else:
                    _log(logger, f"Warning: shot_cut produced no metadata for {video_id}")
            except Exception as exc:
                _log(logger, f"Error processing shot_cut video {video_id}: {exc}. Will retry this video on next run.")
                continue

    _log(logger, f"[PIPELINE] stage=shot_cut status=stage_completed")

    videos_metadata_path = metadata_paths(output_dir, "placeholder", out_format, layout)["videos"]
    if all_videos_metadata:
        save_metadata(all_videos_metadata, videos_metadata_path, out_format)

    free_vram()
    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "processed_videos": len(videos),
        "metadata_files": {"videos": videos_metadata_path},
    }
