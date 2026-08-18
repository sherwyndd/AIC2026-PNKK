import logging
import re
from pathlib import Path

PIPELINE_LOG_PATTERN = re.compile(
    r"\[PIPELINE\] stage=(?P<stage>[a-z_]+) status=(?P<status>[a-z_]+)(?:\s+video_id=(?P<video_id>\S+))?"
)


def setup_logger(log_file=None, level=logging.INFO):
    logger = logging.getLogger("aic_pipeline")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def parse_pipeline_log(log_file):
    stage_state = {
        "shot_cut": False,
        "keyframe_ext": False,
        "asr_whisper": False,
        "object_detection": False,
        "captioning": False,
    }

    if not log_file:
        return stage_state

    try:
        with open(log_file, encoding="utf-8") as handle:
            for line in handle:
                match = PIPELINE_LOG_PATTERN.search(line)
                if not match:
                    continue
                stage = match.group("stage")
                status = match.group("status")
                if status == "stage_completed":
                    stage_state[stage] = True
    except FileNotFoundError:
        pass

    return stage_state
