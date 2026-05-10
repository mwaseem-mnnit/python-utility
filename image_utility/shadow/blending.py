"""Composite shadow into composed RGB without darkening the product."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import ShadowConfig

FloatMask = NDArray[np.float32]
UInt8RGB = NDArray[np.uint8]
UInt8Alpha = NDArray[np.uint8]


def blend_shadow_into_rgb(
    rgb: UInt8RGB,
    product_alpha: UInt8Alpha,
    shadow_blur: FloatMask,
    cfg: ShadowConfig,
) -> UInt8RGB:
    """
    Darken background under ``shadow_blur``; preserve pixels where product is opaque.

    ``product_alpha`` is H×W uint8 aligned with ``rgb``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OSError("rgb must be H×W×3")
    if product_alpha.shape[:2] != rgb.shape[:2] or shadow_blur.shape[:2] != rgb.shape[:2]:
        raise OSError("shadow/product alpha must match rgb shape")

    a = np.clip(product_alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    bg_weight = np.clip(1.0 - a[..., None], 0.0, 1.0)

    strength = float(np.clip(cfg.shadow_opacity * cfg.shadow_intensity, 0.0, 1.0))
    factor = 1.0 - strength * shadow_blur[..., None] * bg_weight

    out = np.clip(rgb.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
    return out
