"""Subtle clarity via LAB lightness separation (avoids RGB color halos)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

UInt8RGB = NDArray[np.uint8]


def clarity_lab_rgb(rgb: UInt8RGB, strength: float, sigma: float) -> UInt8RGB:
    """
    Boost high-frequency detail on the L channel only.

    ``strength <= 0`` or ``sigma <= 0`` returns ``rgb`` unchanged.
    """
    if strength <= 0 or sigma <= 0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_f = l_ch.astype(np.float32)
    blur = cv2.GaussianBlur(l_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    l2 = np.clip(l_f + strength * (l_f - blur), 0.0, 255.0).astype(np.uint8)
    merged = cv2.merge([l2, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
