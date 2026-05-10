"""Mild unsharp masking (RGB)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

UInt8RGB = NDArray[np.uint8]


def unsharp_mask_rgb(rgb: UInt8RGB, strength: float, sigma: float) -> UInt8RGB:
    """
    Conservative unsharp mask: ``out = rgb * (1+s) - blur(rgb) * s``.

    ``strength <= 0`` or ``sigma <= 0`` returns ``rgb`` unchanged.
    """
    if strength <= 0 or sigma <= 0:
        return rgb
    rgb_f = rgb.astype(np.float32)
    blur = cv2.GaussianBlur(rgb_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = rgb_f * (1.0 + strength) - blur * strength
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
