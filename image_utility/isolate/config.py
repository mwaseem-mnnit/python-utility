"""Tunable isolate-phase parameters (env + defaults)."""

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
class IsolateConfig:
    # Alpha / mask
    alpha_visibility_threshold: int
    """Pixels with alpha <= this are treated as transparent for CC / masking."""

    min_component_area: int
    """Discard connected components smaller than this (pixels)."""

    min_fragment_area_after_select: int
    """Remove tiny islands inside the kept mask after main selection."""

    # Morphology (0 = disabled for open; close uses small kernel if > 0)
    morph_pre_close_size: int
    """Odd kernel size for pre-CC closing (fill speckle holes); 0 disables."""

    morph_post_open_size: int
    """Odd kernel size for light opening after selection; 0 disables."""

    # Scoring heuristics (not largest-component)
    center_bias: float
    """Higher → stronger preference for components near image center."""

    complexity_weight: float
    """Weight for contour complexity (structure) in the score."""

    aspect_ratio_penalty_threshold: float
    """If max(w,h)/min(w,h) exceeds this, apply ``elongation_penalty``."""

    elongation_penalty: float
    """Multiply score by this when bbox is very elongated (e.g. finger-like)."""

    # Edge refinement
    edge_blur_sigma: float
    """Gaussian sigma for mild alpha feathering; 0 disables."""

    rgb_zero_below_alpha: int
    """RGB forced to 0 where alpha <= this (reduces matte / fringe)."""

    # rembg
    rembg_model_name: str | None
    """Optional model name for ``rembg.new_session``; None uses default."""

    # Debug
    debug_enabled: bool
    """Write artifacts under ``debug/isolate/`` when True."""


def load_isolate_config() -> IsolateConfig:
    """Load configuration from ``IMAGE_UTIL_ISOLATE_*`` env vars (see defaults)."""

    return IsolateConfig(
        alpha_visibility_threshold=_int_env("IMAGE_UTIL_ISOLATE_ALPHA_THRESH", 8),
        min_component_area=_int_env("IMAGE_UTIL_ISOLATE_MIN_COMPONENT_AREA", 400),
        min_fragment_area_after_select=_int_env("IMAGE_UTIL_ISOLATE_MIN_FRAGMENT_AREA", 120),
        morph_pre_close_size=_int_env("IMAGE_UTIL_ISOLATE_MORPH_PRE_CLOSE", 3),
        morph_post_open_size=_int_env("IMAGE_UTIL_ISOLATE_MORPH_POST_OPEN", 0),
        center_bias=_float_env("IMAGE_UTIL_ISOLATE_CENTER_BIAS", 2.0),
        complexity_weight=_float_env("IMAGE_UTIL_ISOLATE_COMPLEXITY_WEIGHT", 0.15),
        aspect_ratio_penalty_threshold=_float_env("IMAGE_UTIL_ISOLATE_ASPECT_RATIO_THRESH", 4.0),
        elongation_penalty=_float_env("IMAGE_UTIL_ISOLATE_ELONGATION_PENALTY", 0.65),
        edge_blur_sigma=_float_env("IMAGE_UTIL_ISOLATE_EDGE_SIGMA", 0.85),
        rgb_zero_below_alpha=_int_env("IMAGE_UTIL_ISOLATE_RGB_ZERO_ALPHA", 8),
        rembg_model_name=os.getenv("IMAGE_UTIL_ISOLATE_REMBG_MODEL", "").strip() or None,
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
    )
