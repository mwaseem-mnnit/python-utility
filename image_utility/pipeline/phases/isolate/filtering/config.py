"""Stage-local filtering configuration."""

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
class FilteringConfig:
    max_image_ratio: float
    border_penalty: float
    min_detail_density: float
    min_focus_score: float
    max_blob_ratio: float
    reject_threshold: float
    conf_floor: float
    eps: float
    overlay_top_n: int
    debug_enabled: bool


def load_filtering_config() -> FilteringConfig:
    """Loads ``IMAGE_UTIL_ISOLATE_FILTER_*`` vars with defaults (never leaves keys undefined logically)."""

    return FilteringConfig(
        max_image_ratio=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MAX_IMAGE_RATIO", 0.82),
        border_penalty=_float_env("IMAGE_UTIL_ISOLATE_FILTER_BORDER_PENALTY", 1.05),
        min_detail_density=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_DETAIL_DENSITY", 0.07),
        min_focus_score=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_FOCUS_SCORE", 0.06),
        max_blob_ratio=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MAX_BLOB_RATIO", 0.42),
        reject_threshold=_float_env("IMAGE_UTIL_ISOLATE_FILTER_REJECT_THRESHOLD", 0.71),
        conf_floor=_float_env("IMAGE_UTIL_ISOLATE_FILTER_CONF_FLOOR", 0.04),
        eps=_float_env("IMAGE_UTIL_ISOLATE_FILTER_EPS", 1e-6),
        overlay_top_n=_int_env("IMAGE_UTIL_ISOLATE_FILTER_OVERLAY_TOP", 18),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
    )


def load_stop_after_aliases() -> str:
    """Support both IMAGE_UTIL_* and legacy ISOLATE_STOP_AFTER_STAGE."""

    raw = (
        os.getenv("IMAGE_UTIL_ISOLATE_STOP_AFTER_STAGE", "").strip().lower()
        or os.getenv("ISOLATE_STOP_AFTER_STAGE", "").strip().lower()
    )
    return raw
