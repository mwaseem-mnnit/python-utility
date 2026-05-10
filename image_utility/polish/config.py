"""Polish phase tunables (subtle ecommerce refinement, env-driven)."""

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
class PolishConfig:
    sharpen_strength: float
    """Unsharp mask blend strength (0 disables)."""

    unsharp_sigma: float
    """Gaussian sigma for unsharp base blur."""

    contrast_factor: float
    """Linear contrast around ``contrast_midpoint`` (1.0 = none)."""

    contrast_midpoint: float
    """Mid-gray anchor for mild contrast."""

    brightness_delta: float
    """Additive offset on RGB (about nothing)."""

    clarity_strength: float
    """LAB lightness high-frequency boost (0 disables)."""

    clarity_sigma: float
    """Gaussian sigma for clarity separation blur on L channel."""

    white_preserve_threshold: int
    """RGB channels >= this in the **input** keep original values (clean white background)."""

    debug_enabled: bool


def load_polish_config() -> PolishConfig:
    return PolishConfig(
        sharpen_strength=_float_env("IMAGE_UTIL_POLISH_SHARPEN", 0.14),
        unsharp_sigma=_float_env("IMAGE_UTIL_POLISH_UNSHARP_SIGMA", 1.0),
        contrast_factor=_float_env("IMAGE_UTIL_POLISH_CONTRAST", 1.05),
        contrast_midpoint=_float_env("IMAGE_UTIL_POLISH_CONTRAST_MID", 128.0),
        brightness_delta=_float_env("IMAGE_UTIL_POLISH_BRIGHTNESS", 3.0),
        clarity_strength=_float_env("IMAGE_UTIL_POLISH_CLARITY", 0.28),
        clarity_sigma=_float_env("IMAGE_UTIL_POLISH_CLARITY_SIGMA", 1.1),
        white_preserve_threshold=_int_env("IMAGE_UTIL_POLISH_WHITE_THRESHOLD", 252),
        debug_enabled=_bool_env("IMAGE_UTIL_POLISH_DEBUG"),
    )
