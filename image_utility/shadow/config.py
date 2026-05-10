"""Shadow phase tunables (env + defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ShadowConfig:
    """Conservative ecommerce grounding shadow."""

    alpha_threshold: int
    """Alpha value treating pixels as part of product footprint for shadow derivation."""

    horizontal_scale: float
    """Widen flattened shadow footprint (>1 stretches horizontally)."""

    vertical_scale: float
    """Compress shadow vertically (<1 flattens contact shadow)."""

    vertical_offset_px: int
    """Base downward shift after scaling (pixels)."""

    vertical_drop_fraction: float
    """Extra downward shift as a fraction of foreground height (grounds shadow under feet)."""

    blur_sigma: float
    """Gaussian sigma for soft falloff."""

    shadow_opacity: float
    """Upper bound for how much shadow darkens background (0–1)."""

    shadow_intensity: float
    """Extra multiplier for conservative depth (0–1, applied with opacity)."""

    debug_enabled: bool


def load_shadow_config() -> ShadowConfig:
    return ShadowConfig(
        alpha_threshold=_int_env("IMAGE_UTIL_SHADOW_ALPHA_THRESH", 8),
        horizontal_scale=_float_env("IMAGE_UTIL_SHADOW_H_SCALE", 1.12),
        vertical_scale=_float_env("IMAGE_UTIL_SHADOW_V_SCALE", 0.52),
        vertical_offset_px=_int_env("IMAGE_UTIL_SHADOW_OFFSET_Y", 12),
        vertical_drop_fraction=_float_env("IMAGE_UTIL_SHADOW_DROP_FRAC", 0.14),
        blur_sigma=_float_env("IMAGE_UTIL_SHADOW_BLUR_SIGMA", 14.0),
        shadow_opacity=_float_env("IMAGE_UTIL_SHADOW_OPACITY", 0.22),
        shadow_intensity=_float_env("IMAGE_UTIL_SHADOW_INTENSITY", 0.55),
        debug_enabled=_bool_env("IMAGE_UTIL_SHADOW_DEBUG"),
    )
