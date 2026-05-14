"""Deterministic overlay helpers for decomposition debug."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

UInt8 = NDArray[np.uint8]
Labels = NDArray[np.int32]


def label_colorize(labels: Labels, *, seed: int) -> UInt8:
    """RGB visualization of integer label map."""
    n = int(labels.max()) + 1
    if n <= 1:
        return np.zeros((*labels.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    lut = rng.integers(32, 255, size=(n, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[labels]


def stack_mask_overlay(
    rgb: UInt8,
    mask: NDArray[np.bool_],
    color_rgb: tuple[int, int, int],
    alpha: float,
) -> UInt8:
    """Alpha blend of a boolean mask onto RGB (RGB uint8 in/out)."""
    out = rgb.astype(np.float32)
    c = np.array(color_rgb, dtype=np.float32)
    m = mask.astype(np.float32)[:, :, None]
    blend = float(max(0.0, min(1.0, alpha)))
    out = out * (1 - m * blend) + c * (m * blend)
    return np.clip(out, 0, 255).astype(np.uint8)
