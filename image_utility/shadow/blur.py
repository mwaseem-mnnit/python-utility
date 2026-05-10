"""Gaussian blur for soft shadow falloff."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

FloatMask = NDArray[np.float32]


def blur_shadow_mask(mask: FloatMask, sigma: float) -> FloatMask:
    """Apply smooth Gaussian blur to shadow layer; ``sigma <= 0`` returns ``mask``."""
    if sigma <= 0:
        return mask
    blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)
