import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSNET_PYTORCH_PATH = PROJECT_ROOT / "models" / "transnetv2" / "inference-pytorch"
if TRANSNET_PYTORCH_PATH.exists():
    sys.path.append(str(TRANSNET_PYTORCH_PATH))

try:
    from transnetv2_pytorch import TransNetV2
except ImportError:
    TransNetV2 = None


def get_video_metadata(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(total_frames / fps, 6) if fps > 0 else 0

    extra_props = {
        "fourcc_code": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "bitrate_kbps": cap.get(cv2.CAP_PROP_BITRATE),
        "backend": cap.getBackendName(),
        "format": int(cap.get(cv2.CAP_PROP_FORMAT)),
        "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
        "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
        "saturation": cap.get(cv2.CAP_PROP_SATURATION),
        "hue": cap.get(cv2.CAP_PROP_HUE),
        "gain": cap.get(cv2.CAP_PROP_GAIN),
        "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
        "iso_speed": cap.get(cv2.CAP_PROP_ISO_SPEED),
        "buffersize": int(cap.get(cv2.CAP_PROP_BUFFERSIZE)),
    }

    fourcc_int = extra_props.pop("fourcc_code", 0)
    if fourcc_int > 0:
        extra_props["fourcc"] = "".join([
            chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)
        ]).strip()

    filtered_extra = {
        k: v for k, v in extra_props.items()
        if v is not None and v != "" and v != 0 and v != 0.0 and v != -1 and v != -1.0
    }
    cap.release()

    metadata = {
        "video_id": video_id,
        "file_name": os.path.basename(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }
    metadata.update(filtered_extra)
    return metadata


def load_transnet_model(device):
    if TransNetV2 is None:
        return None
    try:
        model = TransNetV2(device=device)
        model.eval()
        return model
    except Exception:
        return None


def predict_video(model, video_path, device, total_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    pbar = tqdm(total=total_frames, desc="Extracting frames", leave=False) if total_frames else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (48, 27))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if pbar:
                pbar.update(1)
    finally:
        if pbar:
            pbar.close()
        cap.release()

    if not frames:
        return None

    frames = np.stack(frames)
    no_padded_frames_start = 25
    no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
    total_padded_len = len(frames) + no_padded_frames_start + no_padded_frames_end
    total_steps = (total_padded_len - 100) // 50 + 1

    def input_iterator():
        start_pad = np.tile(frames[0:1], (no_padded_frames_start, 1, 1, 1))
        end_pad = np.tile(frames[-1:], (no_padded_frames_end, 1, 1, 1))
        padded_inputs = np.concatenate([start_pad, frames, end_pad], axis=0)
        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr : ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    predictions = []
    for inp in tqdm(input_iterator(), total=total_steps, desc="Running TransNetV2 inference", leave=False):
        inp_tensor = torch.from_numpy(inp).to(device)
        with torch.no_grad():
            single_frame_pred, all_frames_pred = model(inp_tensor)
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frames_pred = torch.sigmoid(all_frames_pred["many_hot"]).cpu().numpy()
        predictions.append((single_frame_pred[0, 25:75, 0], all_frames_pred[0, 25:75, 0]))

    single_frame_pred = np.concatenate([single_ for single_, _ in predictions])
    all_frames_pred = np.concatenate([all_ for _, all_ in predictions])
    return single_frame_pred[: len(frames)], all_frames_pred[: len(frames)]


def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    if len(predictions) == 0:
        return np.array([[0, 0]], dtype=np.int32)

    predictions = (predictions > threshold).astype(np.uint8)
    scenes = []
    t_prev, start = -1, 0
    last_i = 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i - 1])
        t_prev = t
        last_i = i
    scenes.append([start, last_i])

    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def scenes_to_shots(scenes, video_id, fps):
    shots_data = []
    for idx, scene in enumerate(scenes):
        start_frame = int(scene[0])
        end_frame = int(scene[1])
        shots_data.append({
            "shot_id": f"{video_id}_shot_{idx + 1:03d}",
            "video_id": video_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "pts_start": start_frame / fps if fps > 0 else 0,
            "pts_end": end_frame / fps if fps > 0 else 0,
            "duration_frames": end_frame - start_frame,
        })
    return shots_data


def segment_video(video_path, video_id, model, device):
    metadata = get_video_metadata(video_path, video_id)
    if not metadata:
        return None, []

    if model is None:
        return metadata, []

    single_frame_predictions, _ = predict_video(
        model, video_path, device, total_frames=metadata.get("total_frames")
    )
    if single_frame_predictions is None:
        return metadata, []

    scenes = predictions_to_scenes(single_frame_predictions)
    shots_data = scenes_to_shots(scenes, video_id, metadata["fps"])
    return metadata, shots_data
