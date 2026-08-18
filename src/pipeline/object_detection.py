import os
from collections import Counter

import torch
try:
    from ultralytics import YOLO
except ImportError as imp_exc:
    raise ImportError(
        "ultralytics package not found. Please install it with 'pip install ultralytics'"
    ) from imp_exc

from .config import get_layout, get_output_dir
from .io import ensure_output_dirs, list_videos, load_metadata_table, load_video_list, object_metadata_path, save_metadata
from src.utils.logger import parse_pipeline_log


def _log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def _load_yolo_model(model_id):
    """Load a YOLO model, providing detailed error information.

    Args:
        model_id (str): Path or identifier of the YOLO model.
    Returns:
        YOLO: Loaded model instance.
    Raises:
        RuntimeError: If model loading fails, with original exception details.
    """
    try:
        return YOLO(model_id)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Failed to load YOLO model '{model_id}'. Original error: {exc}\nTraceback:\n{tb}"
        ) from exc


def _extract_labels(result):
    labels = []
    if not hasattr(result, "boxes") or result.boxes is None:
        return labels

    names = getattr(result, "names", {})
    boxes = result.boxes
    classes = getattr(boxes, "cls", None)
    if classes is None:
        return labels

    if hasattr(classes, "cpu"):
        classes = classes.cpu().numpy()
    for c in classes:
        try:
            index = int(c)
        except (TypeError, ValueError):
            index = None
        if index is None:
            labels.append(str(c))
            continue

        if isinstance(names, dict):
            labels.append(names.get(index, str(index)))
        else:
            try:
                labels.append(names[index])
            except Exception:
                labels.append(str(index))

    return labels


def run_object_detection(
    input_dir,
    output_dir=None,
    mode="demo",
    out_format="json",
    yolov8_model_id="yolov8s.pt",
    conf_threshold=0.25,
    iou_threshold=0.45,
    layout=None,
    logger=None,
    resume=True,
    force=False,
    log_file=None,
    video_list_path=None,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = str(get_output_dir(mode, output_dir))
    layout = layout or get_layout(mode)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_output_dirs(output_dir, layout)
    stage_done = parse_pipeline_log(log_file).get("object_detection", False) if resume else False
    video_list = load_video_list(video_list_path) if video_list_path else None
    videos = list_videos(input_dir, mode, video_list)
    pending_videos = list(videos)

    if stage_done and not force and not video_list_path:
        _log(logger, f"Skipping object_detection stage: already completed in log {log_file}")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_yolo_model(yolov8_model_id)
    processed = 0

    for video_file in videos:
        video_id = os.path.splitext(video_file)[0]
        output_path = object_metadata_path(output_dir, video_id, out_format, layout)
        if os.path.exists(output_path) and not force:
            _log(logger, f"Skipping object detection for {video_id}: existing output found at {output_path}")
            continue

        keyframes_dir = os.path.join(output_dir, layout["metadata_dir"], "keyframes")
        keyframes_csv = os.path.join(keyframes_dir, f"{video_id}_keyframes.csv")
        keyframes_json = os.path.join(keyframes_dir, f"{video_id}_keyframes.json")
        keyframes_df = load_metadata_table(keyframes_csv, keyframes_json)
        if keyframes_df is None or keyframes_df.empty:
            _log(logger, f"Warning: No keyframes metadata for {video_id}, skipping object detection.")
            continue

        _log(logger, f"Processing object detection for video {video_id}")
        detections = []
        for row in keyframes_df.to_dict(orient="records"):
            image_path = os.path.join(output_dir, row.get("image_path", ""))
            if not os.path.exists(image_path):
                _log(logger, f"Warning: Missing keyframe image {image_path} for {video_id}")
                continue

            try:
                results = model.predict(
                    source=image_path,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    device=device,
                    verbose=False,
                )
                labels = []
                if results:
                    labels = _extract_labels(results[0])
            except Exception as exc:
                _log(logger, f"Error during object detection for {image_path}: {exc}")
                labels = []

            counts = Counter(labels)
            detections.append(
                {
                    "keyframe_id": row.get("keyframe_id", ""),
                    "detected_objects": " ".join(labels),
                    "unique_objects": sorted(set(labels)),
                    "object_counts": dict(counts),
                }
            )

        save_metadata(detections, output_path, out_format)
        _log(logger, f"Saved object detection metadata for {video_id} to {output_path}")
        processed += 1

    _log(logger, f"[PIPELINE] stage=object_detection status=stage_completed")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "processed_videos": processed,
    }
