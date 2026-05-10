"""Compose pipeline orchestration (ecommerce framing on white canvas)."""

from __future__ import annotations

import logging

import numpy as np

from image_utility.pipeline.context import PipelineContext

from .blending import blend_rgba_on_canvas
from .config import ComposeConfig, load_compose_config
from .debug import write_compose_debug
from .positioning import placement_origin_top_left
from .scaling import (
    crop_rgba_to_bbox,
    max_scale_for_occupancy,
    scale_rgba_uniform,
    visible_foreground_bbox,
)

LOGGER = logging.getLogger(__name__)


def process_compose(
    context: PipelineContext,
    *,
    cfg: ComposeConfig | None = None,
) -> PipelineContext:
    """
    Place isolated ``current_rgba`` on a white canvas; set ``current_image`` to RGB result.

    Keeps ``current_rgba`` from isolate for downstream phases. Sets ``metadata.compose_applied``.
    """
    cfg = cfg or load_compose_config()
    name = context.input_path.name

    rgba = context.current_rgba
    if rgba is None:
        raise OSError("compose requires current_rgba (run isolate first)")

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise OSError("compose expects H×W×4 RGBA")

    alpha = rgba[:, :, 3]

    if not np.any(alpha > cfg.alpha_bbox_threshold):
        raise OSError("compose: empty alpha / no visible foreground")

    bbox = visible_foreground_bbox(alpha, cfg.alpha_bbox_threshold)
    if bbox is None:
        raise OSError("compose: could not compute foreground bounds")

    LOGGER.info("[compose] extracted foreground bounds %s bbox=(%d,%d,%d,%d)", name, *bbox)

    cropped = crop_rgba_to_bbox(rgba, bbox)
    ch, cw = cropped.shape[:2]

    scale = max_scale_for_occupancy(
        cw,
        ch,
        cfg.canvas_width,
        cfg.canvas_height,
        cfg.occupancy_ratio,
    )
    scaled = scale_rgba_uniform(cropped, scale)
    sh, sw = scaled.shape[:2]

    LOGGER.info(
        "[compose] scaled foreground to occupancy target %s size=(%d,%d) scale=%.4f",
        name,
        sw,
        sh,
        scale,
    )

    origin = placement_origin_top_left(
        cfg.canvas_width,
        cfg.canvas_height,
        sw,
        sh,
        cfg,
    )

    try:
        composed = blend_rgba_on_canvas(
            scaled,
            cfg.canvas_height,
            cfg.canvas_width,
            origin,
            cfg.background_rgb,
        )
    except OSError as exc:
        raise OSError(f"compose blend failed: {exc}") from exc

    write_compose_debug(
        cfg,
        stem=context.input_path.stem,
        source_rgba=rgba,
        bbox_xywh=bbox,
        cropped_rgba=cropped,
        scaled_rgba=scaled,
        canvas_hw=(cfg.canvas_height, cfg.canvas_width),
        origin_xy=origin,
    )

    context.current_image = np.ascontiguousarray(composed)
    canvas_rgba = np.zeros((cfg.canvas_height, cfg.canvas_width, 4), dtype=np.uint8)
    ox, oy = origin
    canvas_rgba[oy : oy + sh, ox : ox + sw] = scaled
    context.composed_rgba_canvas = np.ascontiguousarray(canvas_rgba)

    context.metadata["compose_applied"] = True
    context.metadata["write_format"] = "jpeg"
    context.metadata["jpeg_quality"] = cfg.jpeg_quality
    context.debug["compose_canvas"] = (cfg.canvas_width, cfg.canvas_height)
    context.debug["compose_scale"] = float(scale)
    context.debug["compose_origin"] = origin

    LOGGER.info("[compose] composed %s", name)
    return context
