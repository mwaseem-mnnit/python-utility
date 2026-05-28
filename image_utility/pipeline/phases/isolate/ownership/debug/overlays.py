"""Deterministic colour overlays for ownership debug visualisation."""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
UInt8 = NDArray[np.uint8]
# Colour palette keyed by ownership label
LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "product":        (40, 200, 80),    # green
    "support_object": (220, 60, 60),    # red
    "packaging":      (60, 140, 220),   # blue
    "environment":    (200, 160, 40),   # amber
    "uncertain":      (160, 80, 200),   # purple
}
def blend_mask(
    rgb: UInt8,
    mask: NDArray[np.bool_],
    colour_rgb: tuple[int, int, int],
    strength: float,
) -> UInt8:
    out = rgb.astype(np.float32)
    c = np.array(colour_rgb, dtype=np.float32)
    m = mask.astype(np.float32)[:, :, None]
    s = float(max(0.0, min(1.0, strength)))
    out = out * (1 - m * s) + c * (m * s)
    return np.clip(out, 0, 255).astype(np.uint8)
