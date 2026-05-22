"""Stage-local grouping configuration."""

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
class GroupingConfig:
    # Affinity weights (relationship signals → semantic relatedness)
    w_overlap: float
    w_mask_iou: float
    w_centroid: float
    w_bbox_iou: float
    w_containment: float
    w_edge: float
    w_confidence_sim: float
    w_feature_consistency: float
    merge_affinity_threshold: float
    confidence_floor: float
    grouping_ambiguity_ratio: float
    logit_scale: float
    logit_bias: float
    relationship_viz_min_affinity: float
    math_epsilon: float
    debug_enabled: bool
    debug_top_groups: int


def load_grouping_config() -> GroupingConfig:
    """Env prefix ``IMAGE_UTIL_ISOLATE_GROUP_`` (grouping-local only)."""

    return GroupingConfig(
        w_overlap=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_OVERLAP", 1.0),
        w_mask_iou=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_MASK_IOU", 1.0),
        w_centroid=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_CENTROID", 0.85),
        w_bbox_iou=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_BBOX_IOU", 0.75),
        w_containment=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_CONTAIN", 1.1),
        w_edge=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_EDGE", 0.65),
        w_confidence_sim=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_CONF_SIM", 0.9),
        w_feature_consistency=_float_env("IMAGE_UTIL_ISOLATE_GROUP_W_FEATURE", 0.55),
        merge_affinity_threshold=_float_env("IMAGE_UTIL_ISOLATE_GROUP_MERGE_THRESH", 0.52),
        confidence_floor=_float_env("IMAGE_UTIL_ISOLATE_GROUP_CONF_FLOOR", 0.03),
        grouping_ambiguity_ratio=_float_env("IMAGE_UTIL_ISOLATE_GROUP_AMBIG_RATIO", 0.9),
        logit_scale=_float_env("IMAGE_UTIL_ISOLATE_GROUP_LOGIT_SCALE", 2.6),
        logit_bias=_float_env("IMAGE_UTIL_ISOLATE_GROUP_LOGIT_BIAS", 0.0),
        relationship_viz_min_affinity=_float_env("IMAGE_UTIL_ISOLATE_GROUP_REL_VIZ_MIN", 0.35),
        math_epsilon=_float_env("IMAGE_UTIL_ISOLATE_GROUP_EPS", 1e-6),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
        debug_top_groups=_int_env("IMAGE_UTIL_ISOLATE_GROUP_DEBUG_TOP", 8),
    )
