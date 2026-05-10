"""Shadow pipeline orchestration (grounding on white ecommerce canvas)."""

from __future__ import annotations

import logging

import numpy as np

from image_utility.pipeline.context import PipelineContext

from .blur import blur_shadow_mask
from .blending import blend_shadow_into_rgb
from .config import ShadowConfig, load_shadow_config
from .debug import write_shadow_debug
from .mask import build_shadow_mask

LOGGER = logging.getLogger(__name__)


def process_shadow(
    context: PipelineContext,
    *,
    cfg: ShadowConfig | None = None,
) -> PipelineContext:
    """
    Add subtle contact shadow beneath the product on ``current_image``.

    Requires ``composed_rgba_canvas`` (set by compose) for canvas-aligned alpha.
    Preserves ``current_rgba`` from isolate.
    """
    cfg = cfg or load_shadow_config()
    name = context.input_path.name

    if context.current_image is None:
        raise OSError("shadow requires current_image (compose output)")

    canvas_rgba = context.composed_rgba_canvas
    if canvas_rgba is None:
        raise OSError("shadow requires composed_rgba_canvas (run compose before shadow)")

    if canvas_rgba.ndim != 3 or canvas_rgba.shape[2] != 4:
        raise OSError("composed_rgba_canvas must be H×W×4")

    h, w = context.current_image.shape[:2]
    if canvas_rgba.shape[0] != h or canvas_rgba.shape[1] != w:
        raise OSError("composed_rgba_canvas shape must match current_image")

    alpha = canvas_rgba[:, :, 3]
    if not np.any(alpha > cfg.alpha_threshold):
        raise OSError("shadow: invalid or empty alpha on canvas")

    rgb_before = context.current_image

    raw_mask = build_shadow_mask(alpha, cfg)
    LOGGER.info("[shadow] generated foreground shadow %s", name)

    blurred = blur_shadow_mask(raw_mask, cfg.blur_sigma)
    LOGGER.info("[shadow] blurred shadow mask %s", name)

    try:
        rgb_after = blend_shadow_into_rgb(rgb_before, alpha, blurred, cfg)
    except OSError as exc:
        raise OSError(f"shadow blend failed: {exc}") from exc

    write_shadow_debug(
        cfg,
        stem=context.input_path.stem,
        raw_mask=raw_mask,
        blurred=blurred,
        rgb_after=rgb_after,
    )

    context.current_image = np.ascontiguousarray(rgb_after)
    context.metadata["shadow_applied"] = True
    context.debug["shadow_blur_sigma"] = float(cfg.blur_sigma)

    LOGGER.info("[shadow] blended shadow for %s", name)
    return context
