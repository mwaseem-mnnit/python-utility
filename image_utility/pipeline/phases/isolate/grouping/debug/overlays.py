"""Deterministic overlays for grouping debug (stage-local)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

UInt8 = NDArray[np.uint8]


def blend_mask(
    rgb: UInt8,
    mask: NDArray[np.bool_],
    color_rgb: tuple[int, int, int],
    strength: float,
) -> UInt8:
    out = rgb.astype(np.float32)
    c = np.array(color_rgb, dtype=np.float32)
    m = mask.astype(np.float32)[:, :, None]
    s = float(max(0.0, min(1.0, strength)))
    out = out * (1 - m * s) + c * (m * s)
    return np.clip(out, 0, 255).astype(np.uint8)
