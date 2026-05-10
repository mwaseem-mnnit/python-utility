"""Polish pipeline orchestration (subtle RGB refinement after shadow)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext

from .brightness import adjust_brightness_rgb
from .clarity import clarity_lab_rgb
from .config import PolishConfig, load_polish_config
from .contrast import mild_contrast_rgb
from .debug import write_polish_debug
from .sharpen import unsharp_mask_rgb

LOGGER = logging.getLogger(__name__)


def _preserve_near_white(
    original: np.ndarray,
    polished: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Restore pixels that were near-uniform white in ``original`` to avoid graying the background."""
    mask = np.all(original >= threshold, axis=2)
    out = polished.copy()
    out[mask] = original[mask]
    return out


def process_polish(
    context: PipelineContext,
    *,
    cfg: PolishConfig | None = None,
) -> PipelineContext:
    """
    Apply mild contrast, brightness, clarity, and unsharp sharpening to ``current_image`` (RGB).

    Near-white background pixels (from pre-polish) are restored to keep listings clean.
    """
    cfg = cfg or load_polish_config()
    name = context.input_path.name

    rgb = context.current_image
    if rgb is None:
        raise OSError("polish requires current_image (compose / shadow output)")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OSError("polish expects H×W×3 RGB")

    before = np.ascontiguousarray(rgb)
    work = before
    after_contrast: np.ndarray | None = None
    after_sharpen: np.ndarray | None = None

    if abs(cfg.contrast_factor - 1.0) > 1e-6:
        work = mild_contrast_rgb(work, cfg.contrast_factor, cfg.contrast_midpoint)
        LOGGER.info("[polish] balanced contrast %s", name)
        if cfg.debug_enabled:
            after_contrast = work.copy()

    if abs(cfg.brightness_delta) > 1e-6:
        work = adjust_brightness_rgb(work, cfg.brightness_delta)
        LOGGER.info("[polish] adjusted brightness %s", name)

    if cfg.clarity_strength > 0 and cfg.clarity_sigma > 0:
        try:
            work = clarity_lab_rgb(work, cfg.clarity_strength, cfg.clarity_sigma)
        except cv2.error as exc:
            raise OSError(f"polish clarity failed: {exc}") from exc
        LOGGER.info("[polish] applied clarity %s", name)

    if cfg.sharpen_strength > 0 and cfg.unsharp_sigma > 0:
        work = unsharp_mask_rgb(work, cfg.sharpen_strength, cfg.unsharp_sigma)
        LOGGER.info("[polish] applied sharpening %s", name)
        if cfg.debug_enabled:
            after_sharpen = work.copy()

    work = _preserve_near_white(before, work, cfg.white_preserve_threshold)

    write_polish_debug(
        cfg,
        stem=context.input_path.stem,
        before=before,
        after_sharpen=after_sharpen,
        after_contrast=after_contrast,
        final=work,
    )

    context.current_image = np.ascontiguousarray(work)
    context.metadata["polish_applied"] = True
    context.debug["polish_white_threshold"] = cfg.white_preserve_threshold

    LOGGER.info("[polish] polished %s", name)
    return context
