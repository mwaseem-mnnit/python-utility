"""Alpha edge refinement and final RGBA assembly."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

UInt8 = NDArray[np.uint8]
UInt8RGBA = NDArray[np.uint8]


def refine_alpha_soft(alpha: UInt8, sigma: float) -> UInt8:
    """Mild Gaussian smoothing on alpha (feather edges)."""
    if sigma <= 0:
        return alpha
    a = alpha.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def compose_isolated_rgba(
    original_rgba: UInt8RGBA,
    new_alpha: UInt8,
    cfg: IsolateConfig,
) -> UInt8RGBA:
    """Build RGBA with cleaned alpha; zero RGB where alpha is effectively zero."""
    out = np.zeros_like(original_rgba)
    out[:, :, 3] = new_alpha
    rgb = original_rgba[:, :, :3]
    visible = new_alpha[:, :, np.newaxis] > cfg.rgb_zero_below_alpha
    out[:, :, :3] = np.where(visible, rgb, 0).astype(np.uint8)
    return out
