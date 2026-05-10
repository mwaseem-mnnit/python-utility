"""Placement of scaled foreground on the ecommerce canvas."""

from __future__ import annotations

import numpy as np

from .config import ComposeConfig


def placement_origin_top_left(
    canvas_width: int,
    canvas_height: int,
    foreground_width: int,
    foreground_height: int,
    cfg: ComposeConfig,
) -> tuple[int, int]:
    """
    Return ``(x0, y0)`` top-left for the foreground so its center sits at configurable anchors.

    Coordinates are clamped so the full rectangle fits inside the canvas.
    """
    if foreground_width > canvas_width or foreground_height > canvas_height:
        raise OSError("foreground larger than canvas after scaling")

    ax = float(np.clip(cfg.anchor_x_ratio, 0.0, 1.0))
    ay = float(np.clip(cfg.anchor_y_ratio, 0.0, 1.0))

    cx_canvas = int(round(canvas_width * ax))
    cy_canvas = int(round(canvas_height * ay))

    x0 = cx_canvas - foreground_width // 2
    y0 = cy_canvas - foreground_height // 2

    x0 = int(np.clip(x0, 0, canvas_width - foreground_width))
    y0 = int(np.clip(y0, 0, canvas_height - foreground_height))
    return x0, y0
