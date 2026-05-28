"""Stage-local suppression configuration."""

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
class SuppressionConfig:
    w_border: float
    w_elongation: float
    w_weak_semantic: float
    w_loose_affinity: float
    w_secondary_blob: float
    w_thin_bridge: float
    w_bbox_fill_anomaly: float
    elongation_critical: float
    secondary_blob_critical: float
    removal_likelihood_threshold: float
    confidence_floor: float
    math_epsilon: float
    debug_enabled: bool
    debug_survivor_cap: int


def load_suppression_config() -> SuppressionConfig:
    """Env prefix ``IMAGE_UTIL_ISOLATE_SUPPRESS_``."""

    return SuppressionConfig(
        w_border=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_BORDER", 1.15),
        w_elongation=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_ELONG", 0.85),
        w_weak_semantic=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_WEAK_SEM", 0.75),
        w_loose_affinity=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_LOOSE_AFF", 0.65),
        w_secondary_blob=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_SECOND_BLOB", 0.95),
        w_thin_bridge=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_BRIDGE", 0.7),
        w_bbox_fill_anomaly=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_W_BBOX_FILL", 0.55),
        elongation_critical=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_ELONG_CRIT", 6.5),
        secondary_blob_critical=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_SECOND_BLOB_CRIT", 0.38),
        removal_likelihood_threshold=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_REMOVE_THRESH", 0.62),
        confidence_floor=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_CONF_FLOOR", 0.04),
        math_epsilon=_float_env("IMAGE_UTIL_ISOLATE_SUPPRESS_EPS", 1e-6),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
        debug_survivor_cap=_int_env("IMAGE_UTIL_ISOLATE_SUPPRESS_DEBUG_TOP", 8),
    )
