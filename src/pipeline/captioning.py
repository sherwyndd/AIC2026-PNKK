import os
import time
import torch

from .config import get_layout, get_output_dir
from .io import ensure_output_dirs, list_videos, load_metadata_table, load_video_list, caption_metadata_path, save_metadata
from src.utils.logger import parse_pipeline_log


def _log(logger, message):
    if logger:
        logger.info(message)
    else:
        print(message)


def _get_caption_model_for_vram(model_id=None):
    if model_id is not None and model_id not in {"auto", "AUTO"}:
        return model_id

    return "OpenGVLab/InternVL2_5-2B"


def _load_image_text_pipeline(model_id, device):
    from transformers import pipeline

    device_index = 0 if device == "cuda" else -1
    task_names = ["image-to-text", "image-text-to-text"]
    errors = []

    for task_name in task_names:
        try:
            return pipeline(task_name, model=model_id, device=device_index)
        except Exception as exc:  # pragma: no cover - compatibility fallback for different transformers versions
            errors.append(f"{task_name}: {exc}")

    raise RuntimeError(
        f"Unable to initialize any supported image-text pipeline task for model {model_id}. "
        + "; ".join(errors)
    )


def _load_fallback_caption_pipeline(model_id, device):
    fallback_models = [
        "Salesforce/blip-image-captioning-base",
        "microsoft/git-base-coco",
    ]
    if model_id in fallback_models:
        fallback_models = [m for m in fallback_models if m != model_id]

    last_error = None
    for fallback_model_id in fallback_models:
        try:
            return _load_caption_pipeline(fallback_model_id, device)
        except Exception as exc:  # pragma: no cover - fallback branch for incompatible models
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"All captioning fallback models failed. Final error: {last_error}") from last_error
    raise RuntimeError(f"No captioning fallback models available for {model_id}")


def _load_caption_pipeline(model_id, device):
    effective_model_id = _get_caption_model_for_vram(model_id) if model_id in {"auto", "AUTO", None} else model_id
    model_id_lower = effective_model_id.lower()

    # InternVL 2.5 / InternVL family: use the generic multimodal classes supported by recent transformers.
    if "internvl" in model_id_lower:
        try:
            import transformers
            from transformers import AutoProcessor, AutoModel
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            if device == "cuda":
                torch.cuda.empty_cache()

            # Fix: transformers>=4.48 calls .keys() on `all_tied_weights_keys` inside
            # from_pretrained() on sub-models (e.g. InternLM2Model). We patch the base
            # class with a plain empty dict so that:
            #   - .keys() works without AttributeError
            #   - instance assignment (model.all_tied_weights_keys = x) still works
            #     (a plain class attribute, unlike a property, allows instance override)
            # NOTE: We intentionally use a plain dict, NOT a property, to avoid the
            # "property has no setter" error when sub-models try to set the attribute.
            if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
                transformers.PreTrainedModel.all_tied_weights_keys = {}

            # Attempt 4-bit quantization to save VRAM (~2-3 GB) on GPUs like RTX 2080 Ti.
            # BitsAndBytesConfig requires device_map="auto" (not explicit device assignment).
            quantization_config = None
            if device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                except ImportError:
                    pass  # bitsandbytes not installed; fall back to FP16

            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if device == "cuda":
                if quantization_config is not None:
                    # BitsAndBytes requires device_map="auto" for proper layer placement
                    model_kwargs["device_map"] = "auto"
                    model_kwargs["quantization_config"] = quantization_config
                else:
                    # No quantization: pin explicitly to the selected GPU
                    model_kwargs["device_map"] = {"": device}

            # Load the InternVL model via AutoModel with remote code trust
            model = AutoModel.from_pretrained(effective_model_id, **model_kwargs)

            # Load tokenizer; InternVL's official API uses model.chat(tokenizer, image, question).
            # NOTE: use_fast=False is required because sentencepiece>=0.2.0 introduced strict
            # null-byte validation in the Rust fast tokenizer path. InternVL's tokenizer.model
            # contains special tokens with null bytes, causing "piece must not include null character".
            # The slow (Python) tokenizer backend bypasses this validation safely.
            from transformers import AutoTokenizer
            try:
                processor = AutoTokenizer.from_pretrained(
                    effective_model_id,
                    trust_remote_code=True,
                    use_fast=False,
                )
            except Exception as tok_exc:
                raise RuntimeError(
                    f"Failed to load AutoTokenizer for InternVL model {effective_model_id}. "
                    f"Ensure all required model files and dependencies are available. "
                    f"Original error: {tok_exc}"
                ) from tok_exc


            if device != "cuda":
                model = model.to(device)
            return {"type": "internvl", "model": model, "processor": processor, "device": device}
        except Exception as exc:
            fallback_hint = (
                "InternVL tokenizer initialization failed due a null-byte / sentencepiece incompatibility "
                "in the downloaded model files. Falling back to a known-good image-captioning model."
            )
            try:
                return _load_fallback_caption_pipeline(effective_model_id, device)
            except RuntimeError as fallback_exc:
                raise RuntimeError(
                    f"Failed to initialize InternVL captioning model {effective_model_id}: {exc}. {fallback_hint} "
                    f"Fallback also failed: {fallback_exc}"
                ) from exc

    # Qwen2.5-VL requires its own processor + model class, not the generic pipeline()
    if "Qwen2.5-VL" in effective_model_id or "Qwen2-VL" in effective_model_id:
        try:
            from transformers import AutoProcessor
            dtype = torch.float16 if device == "cuda" else torch.float32
            if device == "cuda":
                torch.cuda.empty_cache()

            # Qwen2.5-VL (v2.5) uses Qwen2_5_VLForConditionalGeneration.
            # Qwen2-VL  (v2.0) uses Qwen2VLForConditionalGeneration.
            # Using the wrong class causes architecture mismatch warnings and OOM.
            if "Qwen2.5-VL" in effective_model_id:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    model_cls = Qwen2_5_VLForConditionalGeneration
                except ImportError:
                    # Older transformers (<4.52) may not have this class yet
                    from transformers import Qwen2VLForConditionalGeneration
                    model_cls = Qwen2VLForConditionalGeneration
            else:
                from transformers import Qwen2VLForConditionalGeneration
                model_cls = Qwen2VLForConditionalGeneration

            quantization_config = None
            if device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                except ImportError:
                    pass

            model = model_cls.from_pretrained(
                effective_model_id,
                dtype=dtype,
                device_map={"": device} if device == "cuda" else None,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
            )
            if device != "cuda":
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(effective_model_id)
            return {"type": "qwen2vl", "model": model, "processor": processor, "device": device}
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Qwen2VL captioning model {effective_model_id}: {exc}"
            ) from exc
    # Generic image-to-text pipeline for other models.
    # Newer transformers versions renamed the task to "image-text-to-text" while
    # older releases still accept "image-to-text". Try both to keep the fallback
    # compatible across different installed environments.
    try:
        return {"type": "pipeline", "pipe": _load_image_text_pipeline(effective_model_id, device)}
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize captioning pipeline for model {effective_model_id}: {exc}"
        ) from exc


def _coerce_caption_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _coerce_caption_text(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("generated_text", "caption", "text", "answer", "response"):
            if key in value:
                text = _coerce_caption_text(value[key])
                if text:
                    return text
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_text_for_comparison(text):
    import re

    if not text:
        return ""
    normalized = re.sub(r"\W+", " ", str(text)).strip().lower()
    return normalized


def _get_resume_start_index(videos, output_dir, out_format, layout, force=False):
    if force:
        return 0

    metadata_dir = os.path.join(output_dir, layout["metadata_dir"])
    keyframes_dir = os.path.join(metadata_dir, "keyframes")
    captions_dir = os.path.join(metadata_dir, "captions")

    max_completed_index = -1
    for index, video_file in enumerate(videos):
        video_id = os.path.splitext(video_file)[0]
        caption_path = os.path.join(captions_dir, f"{video_id}_captions.{out_format}")
        keyframes_path = os.path.join(keyframes_dir, f"{video_id}_keyframes.{out_format}")
        if os.path.exists(caption_path) and os.path.exists(keyframes_path):
            max_completed_index = index

    return max_completed_index + 1


def _is_cuda_oom_error(exc):
    if exc is None:
        return False
    message = str(exc).lower()
    return "cuda out of memory" in message or ("out of memory" in message and "cuda" in message)


def _get_free_gpu_memory_mib():
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024))
    except Exception:
        return None


def _wait_for_gpu_memory(min_free_mem_mib, logger, wait_interval_sec=15):
    while True:
        free_mem_mib = _get_free_gpu_memory_mib()
        if free_mem_mib is None:
            return False
        if free_mem_mib >= min_free_mem_mib:
            return True
        _log(logger, f"GPU memory low ({free_mem_mib} MiB < {min_free_mem_mib} MiB); waiting {wait_interval_sec}s before retrying captioning.")
        time.sleep(wait_interval_sec)


def _is_prompt_echo(generated_text, prompt):
    if not generated_text or not prompt:
        return False
    import re
    from difflib import SequenceMatcher

    generated_norm = _normalize_text_for_comparison(generated_text)
    prompt_norm = _normalize_text_for_comparison(prompt)
    if generated_norm == prompt_norm:
        return True

    if generated_norm.startswith(prompt_norm):
        remainder = generated_norm[len(prompt_norm) :].strip()
        if not remainder:
            return True
        if remainder in {"s", "s.", ".", "?", "!"}:
            return True
        if len(remainder.split()) <= 2 and re.fullmatch(r"[a-z0-9\s.,!?]*", remainder):
            return True
        return False

    similarity = SequenceMatcher(None, generated_norm, prompt_norm).ratio()
    return similarity > 0.95


def _build_caption_prompt_candidates(prompt):
    default_prompt = (
        "Task: Provide a short visual description of the keyframe in 1 sentence (max 20 words). "
        "Instruction: Output ONLY the description. Start directly with \"A \" or \"The \". "
        "Do not guess names, dates, or cities."
    )
    if prompt:
        normalized_prompt = _normalize_text_for_comparison(prompt)
        if normalized_prompt == _normalize_text_for_comparison(default_prompt):
            return ["Describe this image", prompt]
        return [prompt, "Describe this image"]
    return ["Describe this image"]


def _extract_caption(captioner, image_path, prompt):
    from PIL import Image

    if captioner.get("type") == "internvl":
        model = captioner["model"]
        tokenizer = captioner["processor"]
        image = Image.open(image_path).convert("RGB")
        prompt_candidates = _build_caption_prompt_candidates(prompt)
        responses = []
        with torch.no_grad():
            for candidate in prompt_candidates:
                text_prompt = candidate
                if "<image>" not in text_prompt:
                    question = f"<image>\n{text_prompt}"
                else:
                    question = text_prompt

                for call in (
                    lambda: model.chat(tokenizer=tokenizer, image=image, question=question, generation_config={"max_new_tokens": 128, "max_length": None}),
                    lambda: model.chat(tokenizer, image, question, generation_config={"max_new_tokens": 128, "max_length": None}),
                    lambda: model.chat(tokenizer, image, question),
                ):
                    try:
                        responses.append((candidate, call()))
                    except TypeError:
                        continue
                    except Exception:
                        continue

        for candidate, response in responses:
            text = _coerce_caption_text(response)
            if text and not _is_prompt_echo(text, candidate):
                return text

        return ""

    if captioner.get("type") == "qwen2vl":
        model = captioner["model"]
        processor = captioner["processor"]
        device = captioner["device"]
        image = Image.open(image_path).convert("RGB")
        text_prompt = prompt or "Describe this image concisely in English, focusing on key objects, actions, text, and context."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
        except ImportError:
            # Fallback without qwen_vl_utils
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(device)
        generation_config = dict(
            max_new_tokens=40,
            max_length=None,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip() if output_text else ""

    # Generic pipeline path
    pipe = captioner["pipe"]
    prompt_candidates = _build_caption_prompt_candidates(prompt)
    last_caption = ""
    generation_config = dict(
        max_new_tokens=40,
        max_length=None,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    for candidate in prompt_candidates:
        kwargs = {"images": image_path, **generation_config}
        if candidate:
            kwargs["text"] = candidate
        try:
            result = pipe(**kwargs)
        except TypeError:
            # Older transformers pipelines may still expect `inputs` / `prompt` parameters.
            kwargs.pop("text", None)
            kwargs["inputs"] = image_path
            if candidate:
                kwargs["prompt"] = candidate
            result = pipe(**kwargs)

        if isinstance(result, list):
            result = result[0] if result else {}
        caption = _coerce_caption_text(result)
        if caption:
            if not _is_prompt_echo(caption, candidate):
                return caption.strip()
            last_caption = caption.strip()

    return last_caption.strip()


def run_captioning(
    input_dir,
    output_dir=None,
    mode="demo",
    out_format="json",
    caption_model_id="auto",
    caption_prompt=None,
    layout=None,
    logger=None,
    resume=True,
    force=False,
    log_file=None,
    video_list_path=None,
    caption_wait_interval_sec=15,
    caption_min_free_mem_mib=8000,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = str(get_output_dir(mode, output_dir))
    layout = layout or get_layout(mode)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ensure_output_dirs(output_dir, layout)
    stage_done = parse_pipeline_log(log_file).get("captioning", False) if resume else False
    if stage_done and not force and not video_list_path:
        _log(logger, f"Skipping captioning stage: already completed in log {log_file}")
        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "processed_videos": 0,
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_model_id = _get_caption_model_for_vram(caption_model_id)
    captioner = _load_caption_pipeline(selected_model_id, device)
    video_list = load_video_list(video_list_path) if video_list_path else None
    videos = list_videos(input_dir, mode, video_list)
    processed = 0
    start_index = _get_resume_start_index(videos, output_dir, out_format, layout, force=force)
    pending_videos = list(videos[start_index:])
    if start_index > 0:
        previous_video = videos[start_index - 1] if start_index > 0 else None
        _log(logger, f"Resuming captioning from index {start_index}; furthest completed video: {previous_video if previous_video else 'none'}")
    elif videos:
        _log(logger, f"Starting captioning from the beginning with {len(videos)} video(s)")

    while pending_videos:
        deferred = []

        for video_file in pending_videos:
            video_id = os.path.splitext(video_file)[0]
            output_path = caption_metadata_path(output_dir, video_id, out_format, layout)
            if os.path.exists(output_path) and not force:
                _log(logger, f"Skipping captioning for {video_id}: existing output found at {output_path}")
                continue

            keyframes_dir = os.path.join(output_dir, layout["metadata_dir"], "keyframes")
            keyframes_csv = os.path.join(keyframes_dir, f"{video_id}_keyframes.csv")
            keyframes_json = os.path.join(keyframes_dir, f"{video_id}_keyframes.json")
            keyframes_df = load_metadata_table(keyframes_csv, keyframes_json)
            if keyframes_df is None or keyframes_df.empty:
                _log(logger, f"Warning: No keyframes metadata for {video_id}, skipping captioning.")
                continue

            _log(logger, f"Processing captioning for video {video_id} using model {selected_model_id}")
            captions = []
            video_failed_due_to_oom = False
            for row in keyframes_df.to_dict(orient="records"):
                image_path = os.path.join(output_dir, row.get("image_path", ""))
                if not os.path.exists(image_path):
                    _log(logger, f"Warning: Missing keyframe image {image_path} for {video_id}")
                    continue

                try:
                    caption_text = _extract_caption(captioner, image_path, caption_prompt)
                except Exception as exc:
                    if _is_cuda_oom_error(exc):
                        video_failed_due_to_oom = True
                        _log(logger, f"CUDA OOM while captioning {video_id}; pausing and retrying later.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        break
                    _log(logger, f"Error during captioning for {image_path}: {exc}")
                    caption_text = ""

                captions.append(
                    {
                        "keyframe_id": row.get("keyframe_id", ""),
                        "model_id": selected_model_id,
                        "prompt_used": caption_prompt or "",
                        "caption_en": caption_text,
                        "has_caption": bool(caption_text.strip()),
                    }
                )

            if video_failed_due_to_oom:
                deferred.append(video_file)
                continue

            save_metadata(captions, output_path, out_format)
            _log(logger, f"Saved caption metadata for {video_id} to {output_path}")
            processed += 1

        if deferred:
            _log(logger, f"Captioning paused due to GPU memory pressure; retrying {len(deferred)} video(s) after a short wait.")
            if not _wait_for_gpu_memory(caption_min_free_mem_mib, logger, caption_wait_interval_sec):
                _log(logger, "GPU memory monitoring unavailable; continuing without waiting.")
            pending_videos = deferred
            continue

        break

    _log(logger, f"[PIPELINE] stage=captioning status=stage_completed")

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "processed_videos": processed,
    }
