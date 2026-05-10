"""Build grounding shadow mask from canvas-aligned product alpha."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import ShadowConfig

FloatMask = NDArray[np.float32]


def _footprint_vertical_span(base: FloatMask) -> tuple[int, int]:
    hard = (base > 0.05).astype(np.uint8)
    ys, _ = np.where(hard > 0)
    if ys.size == 0:
        return 0, base.shape[0] - 1
    return int(ys.min()), int(ys.max())


def _shadow_anchor_from_footprint(base: FloatMask) -> tuple[float, float]:
    """Horizontal center + lower-vertical anchor near silhouette bottom."""
    hard = (base > 0.05).astype(np.uint8)
    ys, xs = np.where(hard > 0)
    if ys.size == 0:
        h, w = base.shape[:2]
        return (w - 1) * 0.5, (h - 1) * 0.5
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx = (x0 + x1) * 0.5
    cyy = float(y0) + (y1 - y0) * 0.94
    return float(cx), cyy


def soft_footprint(alpha: NDArray[np.uint8], threshold: int) -> FloatMask:
    """Normalize alpha to ``0..1`` float mask (visible product footprint)."""
    denom = max(255 - threshold, 1)
    return np.clip((alpha.astype(np.float32) - float(threshold)) / float(denom), 0.0, 1.0)


def scale_and_offset_shadow_base(
    base: FloatMask,
    cfg: ShadowConfig,
) -> FloatMask:
    """
    Scale footprint about its contact anchor, then shift downward (contact-style shadow seed).
    """
    h, w = base.shape[:2]
    cx, cyy = _shadow_anchor_from_footprint(base)
    y0, y1 = _footprint_vertical_span(base)
    span = max(1, y1 - y0)
    extra_drop = float(cfg.vertical_drop_fraction) * float(span)

    sx = max(cfg.horizontal_scale, 1e-3)
    sy = max(cfg.vertical_scale, 1e-3)

    M = np.array(
        [
            [sx, 0.0, cx * (1.0 - sx)],
            [0.0, sy, cyy * (1.0 - sy)],
        ],
        dtype=np.float32,
    )
    M[1, 2] += float(cfg.vertical_offset_px) + extra_drop

    warped = cv2.warpAffine(
        base,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped, 0.0, 1.0)


def build_shadow_mask(
    alpha_canvas: NDArray[np.uint8],
    cfg: ShadowConfig,
) -> FloatMask:
    """
    Derive a soft shadow **seed** mask on canvas (before blur), from product alpha.
    """
    if alpha_canvas.ndim != 2:
        raise OSError("alpha canvas must be H×W")
    base = soft_footprint(alpha_canvas, cfg.alpha_threshold)
    if not np.any(base > 1e-3):
        raise OSError("shadow: empty footprint on canvas")
    return scale_and_offset_shadow_base(base, cfg)
