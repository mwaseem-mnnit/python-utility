"""Mild contrast around a midpoint (preserves extremes when factor ≈ 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

UInt8RGB = NDArray[np.uint8]


def mild_contrast_rgb(rgb: UInt8RGB, factor: float, midpoint: float) -> UInt8RGB:
    """``out = (rgb - midpoint) * factor + midpoint`` clipped to uint8."""
    if abs(factor - 1.0) < 1e-6:
        return rgb
    x = rgb.astype(np.float32)
    out = (x - midpoint) * factor + midpoint
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
