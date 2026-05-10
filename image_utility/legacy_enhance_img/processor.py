"""
Image enhancement processor for product photos.

Pipeline:
1. ``extract_foreground`` — rembg segmentation, returns BGRA (with alpha).
2. ``place_on_white_background`` — alpha-blend onto a fixed 2000x1300 canvas.
3. ``polish_image`` — mild brightness, contrast and unsharp-mask polish.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

from image_utility.config import ENV_INPUT_DIR, ENV_OUTPUT_DIR, ENV_MAX_FILES
from image_utility.utils import (
    init_job_logging,
    load_image_utility_env,
    parse_positive_int_env,
    resolve_dir_from_env,
    sorted_image_files,
)

CANVAS_WIDTH = 2000
CANVAS_HEIGHT = 1300
PRODUCT_FILL_RATIO = 0.80
ALPHA_VISIBILITY_THRESHOLD = 8

_SESSION = None


def _rembg_session():
    """Load the rembg model once and reuse across calls in the same process."""
    global _SESSION
    if _SESSION is None:
        _SESSION = new_session()
    return _SESSION


def extract_foreground(image_path: str) -> np.ndarray:
    """
    Run rembg on the input image and return a BGRA ``np.ndarray``.

    The output preserves fine details (edges, wires, curves) via the segmentation
    model's soft alpha mask. Channel order is BGRA for direct OpenCV use.
    """
    with Image.open(image_path) as src:
        rgb_pil = src.convert("RGB")
        rgba_pil = remove(rgb_pil, session=_rembg_session())

    if rgba_pil.mode != "RGBA":
        rgba_pil = rgba_pil.convert("RGBA")
    rgba_arr = np.array(rgba_pil)
    return cv2.cvtColor(rgba_arr, cv2.COLOR_RGBA2BGRA)


def _alpha_bbox(alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    visible = alpha > ALPHA_VISIBILITY_THRESHOLD
    if not visible.any():
        return None
    ys, xs = np.where(visible)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _resize_bgra(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return image
    scale = min(max_width / w, max_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def place_on_white_background(image_rgba: np.ndarray) -> np.ndarray:
    """
    Tightly crop to the alpha bbox, scale to ~80% of the canvas, and alpha-blend
    onto a pure white 2000x1300 canvas. Returns BGR.
    """
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
    if image_rgba.size == 0 or image_rgba.shape[2] != 4:
        return canvas

    alpha = image_rgba[:, :, 3]
    bbox = _alpha_bbox(alpha)
    if bbox is None:
        return canvas

    x0, y0, x1, y1 = bbox
    cropped = image_rgba[y0 : y1 + 1, x0 : x1 + 1]

    max_w = int(CANVAS_WIDTH * PRODUCT_FILL_RATIO)
    max_h = int(CANVAS_HEIGHT * PRODUCT_FILL_RATIO)
    resized = _resize_bgra(cropped, max_w, max_h)

    h, w = resized.shape[:2]
    x = (CANVAS_WIDTH - w) // 2
    y = (CANVAS_HEIGHT - h) // 2

    fg_bgr = resized[:, :, :3].astype(np.float32)
    fg_alpha = (resized[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    bg_region = canvas[y : y + h, x : x + w].astype(np.float32)

    blended = fg_bgr * fg_alpha + bg_region * (1.0 - fg_alpha)
    canvas[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return canvas


def polish_image(image: np.ndarray) -> np.ndarray:
    """
    Apply a mild brightness lift, gentle contrast bump, and unsharp mask.
    Pure white background pixels remain pure white after these operations.
    """
    out = image.astype(np.float32)
    out = out * 1.07 + 4.0
    out = (out - 128.0) * 1.08 + 128.0
    out = np.clip(out, 0, 255).astype(np.uint8)

    smooth = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(out, 1.18, smooth, -0.18, 0)


def process_single_image(input_path: Path, output_path: Path) -> bool:
    """
    Run the full pipeline for one image. Returns True on success.
    Output is written as JPEG (quality 94) regardless of source extension.
    """
    logger = logging.getLogger(__name__)
    try:
        rgba = extract_foreground(str(input_path))
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Foreground extraction failed for %s: %s", input_path.name, exc)
        return False

    canvas = place_on_white_background(rgba)
    polished = polish_image(canvas)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_path = output_path.with_suffix(".jpg")
    ok = cv2.imwrite(str(final_path), polished, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return bool(ok)


def run() -> int:
    """Run the image enhancement job using ``image_utility/.env``."""
    load_image_utility_env()
    init_job_logging("enhance_img.log")
    logger = logging.getLogger(__name__)

    input_dir = resolve_dir_from_env(ENV_INPUT_DIR)
    output_dir = resolve_dir_from_env(ENV_OUTPUT_DIR)
    max_files = parse_positive_int_env(ENV_MAX_FILES)

    if input_dir is None:
        logger.error("%s is not set in .env.", ENV_INPUT_DIR)
        return 1
    if output_dir is None:
        logger.error("%s is not set in .env.", ENV_OUTPUT_DIR)
        return 1
    if not input_dir.is_dir():
        logger.error("Not a directory: %s", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    skipped_count = 0

    for input_path in sorted_image_files(input_dir):
        if max_files is not None and (ok_count + skipped_count) >= max_files:
            logger.info("Reached %s=%s, stopping.", ENV_MAX_FILES, max_files)
            break

        logger.info("Processing %s", input_path.name)
        output_path = output_dir / input_path.name
        try:
            ok = process_single_image(input_path, output_path)
        except OSError as exc:
            logger.warning("Skip %s: %s", input_path.name, exc)
            skipped_count += 1
            continue

        if not ok:
            logger.warning("Skip %s: invalid or corrupt image.", input_path.name)
            skipped_count += 1
            continue

        logger.info("Saved %s", input_path.name)
        ok_count += 1

    logger.info("Done. %s image(s) processed, %s skipped.", ok_count, skipped_count)
    return 0
