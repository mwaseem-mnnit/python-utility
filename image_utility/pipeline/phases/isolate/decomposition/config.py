"""Stage-local decomposition configuration (env-driven)."""

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


def _str_opt(key: str) -> str | None:
    raw = os.getenv(key, "").strip()
    return raw or None


@dataclass(frozen=True)
class DecompositionConfig:
    alpha_visibility_threshold: int
    morph_pre_close_size: int
    morph_post_open_size: int
    rembg_model_name: str | None
    sam_enabled: bool
    sam_checkpoint: str | None
    sam_use_gpu: bool
    sam_points_per_side: int
    sam_pred_iou_thresh: float
    sam_stability_thresh: float
    sam_min_mask_region_area: int
    candidate_max_regions: int
    candidate_overlap_dedup_threshold: float
    candidate_dedup_min_area_ratio: float
    min_region_area_cc: int
    normalize_min_mask_area: int
    debug_enabled: bool


def load_decomposition_config() -> DecompositionConfig:
    """Env prefix ``IMAGE_UTIL_ISOLATE_DECOMP_`` for stage-local knobs."""

    return DecompositionConfig(
        alpha_visibility_threshold=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_ALPHA_THRESH", 8),
        morph_pre_close_size=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_MORPH_PRE_CLOSE", 3),
        morph_post_open_size=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_MORPH_POST_OPEN", 0),
        rembg_model_name=_str_opt("IMAGE_UTIL_ISOLATE_DECOMP_REMBG_MODEL")
        or _str_opt("IMAGE_UTIL_ISOLATE_REMBG_MODEL"),
        sam_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM")
        or _bool_env("IMAGE_UTIL_ISOLATE_SEMANTIC"),
        sam_checkpoint=_str_opt("IMAGE_UTIL_ISOLATE_DECOMP_SAM_CHECKPOINT")
        or _str_opt("IMAGE_UTIL_ISOLATE_SAM_CHECKPOINT"),
        sam_use_gpu=_bool_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM_GPU")
        or _bool_env("IMAGE_UTIL_ISOLATE_SAM_GPU"),
        sam_points_per_side=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM_POINTS", 12),
        sam_pred_iou_thresh=_float_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM_PRED_IOU", 0.78),
        sam_stability_thresh=_float_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM_STABILITY", 0.88),
        sam_min_mask_region_area=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_SAM_MIN_AREA", 120),
        candidate_max_regions=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_MAX_REGIONS", 256),
        candidate_overlap_dedup_threshold=_float_env("IMAGE_UTIL_ISOLATE_DECOMP_DEDUP_IOU", 0.92),
        candidate_dedup_min_area_ratio=_float_env("IMAGE_UTIL_ISOLATE_DECOMP_DEDUP_AREA_RATIO", 0.88),
        min_region_area_cc=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_MIN_CC_AREA", 50),
        normalize_min_mask_area=_int_env("IMAGE_UTIL_ISOLATE_DECOMP_NORM_MIN_AREA", 64),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
    )
