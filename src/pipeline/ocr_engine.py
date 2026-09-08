import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
from tqdm import tqdm


def parse_page_result(res: Any, conf_threshold: float = 0.5, scale_factor: float = 1.0) -> Tuple[List[Dict[str, Any]], str]:
    """Extract texts, confidence scores, and bounding polygons from a PaddleOCR result."""
    texts = []
    merged_text = []

    if not res:
        return texts, ""

    page = res[0] if (isinstance(res, list) and len(res) > 0) else (res if isinstance(res, dict) else {})
    if not isinstance(page, dict):
        return texts, ""

    rec_texts = page.get("rec_texts", [])
    rec_scores = page.get("rec_scores", [])
    dt_polys = page.get("dt_polys", [])

    for poly, text, score in zip(dt_polys, rec_texts, rec_scores):
        text = str(text).strip()
        score = float(score)

        if not text or score < conf_threshold:
            continue

        poly_list = poly.tolist() if hasattr(poly, "tolist") else poly

        if scale_factor != 1.0 and poly_list:
            inv_scale = 1.0 / scale_factor
            poly_list = [[pt[0] * inv_scale, pt[1] * inv_scale] for pt in poly_list]

        texts.append({
            "text": text,
            "score": score,
            "polygon": poly_list,
        })
        merged_text.append(text)

    return texts, " ".join(merged_text)


class OCREngine:
    """Core OCR Engine wrapping PaddleOCR with batching and memory cleanup."""

    def __init__(self, lang: str = "vi", use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr = None

    def _init_ocr(self):
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang=self.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            return self._ocr
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed in the active Python environment. "
                "Please install paddleocr / paddlepaddle or run via an OCR-enabled environment."
            ) from exc

    def clear_memory(self):
        """Clear GPU memory caches."""
        gc.collect()
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def process_keyframe_dir(
        self,
        keyframe_dir: Path,
        output_file: Path,
        conf_threshold: float = 0.5,
        batch_size: int = 16,
        logger: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Run OCR on all keyframe images in keyframe_dir and save to output_file JSON."""
        ocr = self._init_ocr()
        image_paths = sorted(Path(keyframe_dir).rglob("*.jpg"))
        if not image_paths:
            image_paths = sorted(Path(keyframe_dir).rglob("*.png"))

        if not image_paths:
            if logger:
                logger.info(f"No keyframe images found in {keyframe_dir}")
            return []

        results = []
        i = 0
        pbar = tqdm(total=len(image_paths), desc=f"OCR {keyframe_dir.name}")

        while i < len(image_paths):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            metadata = []

            for path in batch_paths:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                images.append(img)
                metadata.append((str(path), h, w))

            if not images:
                i += len(batch_paths)
                pbar.update(len(batch_paths))
                continue

            try:
                if len(images) == 1:
                    batch_res = [ocr.predict(images[0])]
                else:
                    batch_res = ocr.predict(images)

                if not isinstance(batch_res, list):
                    batch_res = [batch_res]

                for k, (rel_path, h, w) in enumerate(metadata):
                    res = batch_res[k] if k < len(batch_res) else None
                    texts, merged_str = parse_page_result(res, conf_threshold)
                    results.append({
                        "doc_id": len(results),
                        "image": rel_path,
                        "width": w,
                        "height": h,
                        "texts": texts,
                        "text": merged_str,
                    })

                i += len(batch_paths)
                pbar.update(len(batch_paths))
            except Exception as exc:
                if logger:
                    logger.error(f"Error processing OCR batch at index {i}: {exc}")
                i += len(batch_paths)
                pbar.update(len(batch_paths))

        pbar.close()

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.clear_memory()
        return results
