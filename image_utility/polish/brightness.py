"""Subtle brightness offset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

UInt8RGB = NDArray[np.uint8]


def adjust_brightness_rgb(rgb: UInt8RGB, delta: float) -> UInt8RGB:
    """Add ``delta`` to all channels (use small values for ecommerce subtlety)."""
    if abs(delta) < 1e-6:
        return rgb
    return np.clip(rgb.astype(np.float32) + delta, 0.0, 255.0).astype(np.uint8)
