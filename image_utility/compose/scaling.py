"""Foreground bounds, crop-to-alpha, and proportional scaling."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

UInt8RGBA = NDArray[np.uint8]


def visible_foreground_bbox(alpha: NDArray[np.uint8], threshold: int) -> tuple[int, int, int, int] | None:
    """
    Tight axis-aligned bbox (x, y, w, h) of pixels with ``alpha > threshold``.

    Returns ``None`` if no visible pixels.
    """
    if alpha.ndim != 2:
        raise OSError("alpha must be H×W")
    vis = alpha > threshold
    if not np.any(vis):
        return None
    ys, xs = np.where(vis)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def crop_rgba_to_bbox(rgba: UInt8RGBA, bbox: tuple[int, int, int, int]) -> UInt8RGBA:
    """Crop RGBA to ``(x, y, w, h)`` in image coordinates; preserves alpha."""
    x, y, w, h = bbox
    ih, iw = rgba.shape[:2]
    if x < 0 or y < 0 or x + w > iw or y + h > ih:
        raise OSError("bbox outside image bounds")
    if w < 1 or h < 1:
        raise OSError("invalid bbox dimensions")
    return np.ascontiguousarray(rgba[y : y + h, x : x + w])


def max_scale_for_occupancy(
    fg_width: int,
    fg_height: int,
    canvas_width: int,
    canvas_height: int,
    occupancy_ratio: float,
) -> float:
    """Uniform scale so foreground fits within occupancy box on canvas."""
    if fg_width < 1 or fg_height < 1:
        raise OSError("invalid foreground size")
    max_w = max(1.0, float(canvas_width) * occupancy_ratio)
    max_h = max(1.0, float(canvas_height) * occupancy_ratio)
    return float(min(max_w / fg_width, max_h / fg_height))


def scale_rgba_uniform(rgba: UInt8RGBA, scale: float) -> UInt8RGBA:
    """Resize RGBA preserving aspect ratio (``scale`` applied to width and height)."""
    if scale <= 0:
        raise OSError("scale must be positive")
    h, w = rgba.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(rgba, (new_w, new_h), interpolation=interp)
