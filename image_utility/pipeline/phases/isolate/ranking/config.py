"""Stage-local ranking configuration."""

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
class RankingConfig:
    alpha_visibility_threshold: int
    center_bias: float
    center_score_weight: float
    border_penalty_weight: float
    border_gamma: float
    solidity_weight: float
    solidity_floor: float
    elongation_thresh: float
    elongation_penalty_weight: float
    complexity_weight: float
    complexity_cap: float
    area_log_weight: float
    sam_iou_weight: float
    sam_stability_weight: float
    overlap_rembg_weight: float
    logit_bias: float
    logit_scale: float
    confidence_floor: float
    ambiguity_ratio_threshold: float
    math_epsilon: float
    debug_enabled: bool


def load_ranking_config() -> RankingConfig:
    """Env prefix ``IMAGE_UTIL_ISOLATE_RANK_``."""

    return RankingConfig(
        alpha_visibility_threshold=_int_env("IMAGE_UTIL_ISOLATE_RANK_ALPHA_THRESH", 8),
        center_bias=_float_env("IMAGE_UTIL_ISOLATE_RANK_CENTER_BIAS", 2.0),
        center_score_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_CENTER", 1.15),
        border_penalty_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_BORDER", 1.85),
        border_gamma=_float_env("IMAGE_UTIL_ISOLATE_RANK_BORDER_GAMMA", 1.1),
        solidity_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_SOLIDITY", 0.55),
        solidity_floor=_float_env("IMAGE_UTIL_ISOLATE_RANK_SOLIDITY_FLOOR", 0.28),
        elongation_thresh=_float_env("IMAGE_UTIL_ISOLATE_RANK_ELONG_THRESH", 4.0),
        elongation_penalty_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_ELONG", 0.35),
        complexity_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_COMPLEXITY", 0.45),
        complexity_cap=_float_env("IMAGE_UTIL_ISOLATE_RANK_COMPLEXITY_CAP", 10.0),
        area_log_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_AREA", 1.0),
        sam_iou_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_SAM_IOU", 0.75),
        sam_stability_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_SAM_STAB", 0.4),
        overlap_rembg_weight=_float_env("IMAGE_UTIL_ISOLATE_RANK_W_OVERLAP", 0.9),
        logit_bias=_float_env("IMAGE_UTIL_ISOLATE_RANK_LOGIT_BIAS", -0.1),
        logit_scale=_float_env("IMAGE_UTIL_ISOLATE_RANK_LOGIT_SCALE", 1.0),
        confidence_floor=_float_env("IMAGE_UTIL_ISOLATE_RANK_CONF_FLOOR", 0.02),
        ambiguity_ratio_threshold=_float_env("IMAGE_UTIL_ISOLATE_RANK_AMBIG_RATIO", 0.92),
        math_epsilon=_float_env("IMAGE_UTIL_ISOLATE_RANK_EPS", 1e-6),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
    )
