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

    # ── Deduplication (collapse near-identical SAM proposals pre-scoring) ──
    dedup_iou_threshold: float
    """Mask IoU above which two proposals are considered duplicates (keep larger)."""

    dedup_enabled: bool
    """Enable pre-scoring deduplication."""

    # ── Tiny-area rejection ──
    min_area_pixels: int
    """Proposals below this pixel area are penalised as noise."""

    min_area_ratio: float
    """Proposals with relative area below this are penalised (fraction of image)."""

    # ── Peripheral organic (border-entry + elongation compound) ──
    peripheral_border_min: float
    """Border contact ratio above which peripheral analysis activates."""

    peripheral_elongation_min: float
    """Elongation above which a bordered proposal is flagged as support-like."""

    peripheral_penalty_weight: float
    """Weight of the peripheral+elongation compound penalty in rejection fusion."""

    # ── Low SAM confidence penalty ──
    min_sam_stability: float
    """SAM stability below this penalises the proposal."""

    sam_confidence_penalty_weight: float
    """Weight of the SAM low-confidence penalty in rejection fusion."""

    # ── Coverage (lowered for ecommerce domain) ──
    coverage_penalty_weight: float
    """Direct penalty weight for proposals that cover a large fraction of the image."""

    coverage_soft_start: float
    """Coverage ratio above which a soft penalty starts ramping (before slab threshold)."""


def load_filtering_config() -> FilteringConfig:
    """Loads ``IMAGE_UTIL_ISOLATE_FILTER_*`` vars with defaults (never leaves keys undefined logically)."""

    return FilteringConfig(
        max_image_ratio=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MAX_IMAGE_RATIO", 0.65),
        border_penalty=_float_env("IMAGE_UTIL_ISOLATE_FILTER_BORDER_PENALTY", 1.25),
        min_detail_density=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_DETAIL_DENSITY", 0.07),
        min_focus_score=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_FOCUS_SCORE", 0.06),
        max_blob_ratio=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MAX_BLOB_RATIO", 0.42),
        reject_threshold=_float_env("IMAGE_UTIL_ISOLATE_FILTER_REJECT_THRESHOLD", 0.48),
        conf_floor=_float_env("IMAGE_UTIL_ISOLATE_FILTER_CONF_FLOOR", 0.04),
        eps=_float_env("IMAGE_UTIL_ISOLATE_FILTER_EPS", 1e-6),
        overlay_top_n=_int_env("IMAGE_UTIL_ISOLATE_FILTER_OVERLAY_TOP", 18),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
        # Deduplication
        dedup_iou_threshold=_float_env("IMAGE_UTIL_ISOLATE_FILTER_DEDUP_IOU", 0.85),
        dedup_enabled=not _bool_env("IMAGE_UTIL_ISOLATE_FILTER_DEDUP_DISABLE"),
        # Tiny-area
        min_area_pixels=_int_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_AREA_PX", 300),
        min_area_ratio=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_AREA_RATIO", 0.0008),
        # Peripheral organic
        peripheral_border_min=_float_env("IMAGE_UTIL_ISOLATE_FILTER_PERIPH_BORDER_MIN", 0.005),
        peripheral_elongation_min=_float_env("IMAGE_UTIL_ISOLATE_FILTER_PERIPH_ELONG_MIN", 2.2),
        peripheral_penalty_weight=_float_env("IMAGE_UTIL_ISOLATE_FILTER_PERIPH_WEIGHT", 1.40),
        # Low SAM confidence
        min_sam_stability=_float_env("IMAGE_UTIL_ISOLATE_FILTER_MIN_SAM_STAB", 0.50),
        sam_confidence_penalty_weight=_float_env("IMAGE_UTIL_ISOLATE_FILTER_SAM_CONF_WEIGHT", 0.90),
        # Coverage
        coverage_penalty_weight=_float_env("IMAGE_UTIL_ISOLATE_FILTER_COV_WEIGHT", 1.50),
        coverage_soft_start=_float_env("IMAGE_UTIL_ISOLATE_FILTER_COV_SOFT_START", 0.30),
    )


def load_stop_after_aliases() -> str:
    """Support both IMAGE_UTIL_* and legacy ISOLATE_STOP_AFTER_STAGE."""

    raw = (
        os.getenv("IMAGE_UTIL_ISOLATE_STOP_AFTER_STAGE", "").strip().lower()
        or os.getenv("ISOLATE_STOP_AFTER_STAGE", "").strip().lower()
    )
    return raw
