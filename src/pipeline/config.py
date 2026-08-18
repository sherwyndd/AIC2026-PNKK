from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.yaml"
PIPELINE_TREE_PATH = PROJECT_ROOT / "pipeline_config.yaml"

DEFAULT_INPUT_DIR = "/mlcv2025/Datasets/HCMAI25/batch2/video"
DEFAULT_OUTPUT_DEMO = "dataset/demo"
DEFAULT_OUTPUT_FULL = "data"
DEFAULT_MODE = "demo"
DEFAULT_FORMAT = "csv"
DEFAULT_KEYFRAME_STEP = 8
DEFAULT_REL_DIFF_THRESHOLD = 0.4
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_CLIP_BATCH_SIZE = 8
DEFAULT_YOLOV8_MODEL_ID = "yolov8s.pt"
DEFAULT_OBJECT_DETECTION_CONF = 0.25
DEFAULT_OBJECT_DETECTION_IOU = 0.45
DEFAULT_CAPTION_MODEL_ID = "OpenGVLab/InternVL2_5-2B"
DEFAULT_CAPTION_PROMPT = (
    "Task: Provide a short visual description of the keyframe in 1 sentence (max 20 words). "
    "Instruction: Output ONLY the description. Start directly with \"A \" or \"The \". "
    "Do not guess names, dates, or cities."
)

LAYOUT_DEMO = {"metadata_dir": "Metadata", "keyframes_dir": "Keyframes"}
LAYOUT_FULL = {"metadata_dir": "metadata", "keyframes_dir": "keyframes"}


def load_runtime_config(config_path=None):
    path = Path(config_path) if config_path else RUNTIME_CONFIG_PATH
    if not path.exists():
        return {
            "paths": {
                "input_dir": DEFAULT_INPUT_DIR,
                "output_demo": DEFAULT_OUTPUT_DEMO,
                "output_full": DEFAULT_OUTPUT_FULL,
            },
            "pipeline": {
                "mode": DEFAULT_MODE,
                "format": DEFAULT_FORMAT,
                "sample_step": DEFAULT_KEYFRAME_STEP,
                "rel_diff_threshold": DEFAULT_REL_DIFF_THRESHOLD,
                "clip_batch_size": DEFAULT_CLIP_BATCH_SIZE,
            },
            "models": {"clip_model_id": DEFAULT_CLIP_MODEL_ID},
        }

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def get_output_dir(mode, output_dir=None, config=None):
    if output_dir is not None:
        return resolve_path(output_dir)

    cfg = config or load_runtime_config()
    paths = cfg.get("paths", {})
    rel = paths.get("output_demo" if mode == "demo" else "output_full", DEFAULT_OUTPUT_DEMO)
    return resolve_path(rel)


def get_layout(mode):
    return LAYOUT_DEMO if mode == "demo" else LAYOUT_FULL


def get_pipeline_defaults(config=None):
    cfg = config or load_runtime_config()
    paths = cfg.get("paths", {})
    pipeline = cfg.get("pipeline", {})
    models = cfg.get("models", {})
    return {
        "input_dir": resolve_path(paths.get("input_dir", DEFAULT_INPUT_DIR)),
        "mode": pipeline.get("mode", DEFAULT_MODE),
        "format": pipeline.get("format", DEFAULT_FORMAT),
        "sample_step": pipeline.get("sample_step", DEFAULT_KEYFRAME_STEP),
        "rel_diff_threshold": pipeline.get("rel_diff_threshold", DEFAULT_REL_DIFF_THRESHOLD),
        "clip_batch_size": pipeline.get("clip_batch_size", DEFAULT_CLIP_BATCH_SIZE),
        "clip_model_id": models.get("clip_model_id", DEFAULT_CLIP_MODEL_ID),
        "yolov8_model_id": models.get("yolov8_model_id", DEFAULT_YOLOV8_MODEL_ID),
        "object_detection_confidence": models.get("object_detection_confidence", DEFAULT_OBJECT_DETECTION_CONF),
        "object_detection_iou": models.get("object_detection_iou", DEFAULT_OBJECT_DETECTION_IOU),
        "caption_model_id": models.get("caption_model_id", DEFAULT_CAPTION_MODEL_ID),
        "caption_prompt": models.get("caption_prompt", DEFAULT_CAPTION_PROMPT),
    }
