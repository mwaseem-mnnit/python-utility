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


def _str_env_optional(key: str) -> str | None:
    raw = os.getenv(key, "").strip()
    return raw or None


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

    # Legacy / shared geometry heuristics
    center_bias: float
    """Higher → stronger preference for components near image center."""

    complexity_weight: float
    """Weight for contour complexity inside confidence model."""

    complexity_score_cap: float
    """Upper bound on raw complexity before normalization."""

    aspect_ratio_penalty_threshold: float
    """Elongation above this adds a soft penalty term."""

    elongation_penalty: float
    """Legacy elongation multiplier (unused in v2 logit; kept for env compatibility)."""

    math_epsilon: float
    """Small value to avoid division by zero in normalized distances."""

    # --- Isolate v2: confidence model (weighted logit → sigmoid) ---
    v2_weight_relative_area: float
    """Coefficient for ``log(relative_area)`` term."""

    v2_weight_centrality: float
    """Coefficient for centrality factor (0–1)."""

    v2_weight_complexity_norm: float
    """Coefficient for normalized complexity (0–1)."""

    v2_weight_solidity: float
    """Coefficient for solidity excess above floor (conservative for thin parts)."""

    v2_weight_border_contact: float
    """Penalty weight for ``border_contact_ratio`` (higher = stronger edge aversion)."""

    v2_border_contact_gamma: float
    """Exponent on border ratio (>=1 slightly amplifies strong border contact)."""

    v2_weight_elongation_excess: float
    """Penalty for ``max(0, elongation - threshold)``."""

    v2_confidence_bias: float
    """Added to logit before sigmoid (global calibration)."""

    v2_confidence_scale: float
    """Multiplies the combined score before sigmoid (sharper/softer separation)."""

    v2_min_select_confidence: float
    """If best confidence is below this, selection fails (0–1)."""

    v2_semantic_reject_relative: float
    """Non-winners with confidence below ``best * this`` are labeled *reject*."""

    v2_semantic_uncertain_relative: float
    """Above reject cutoff but below ``best * this`` → *uncertain*."""

    solidity_floor: float
    """Solidity values below this are treated as this floor (protects wires/thin parts)."""

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

    debug_overlay_blend: float
    """Green overlay blend strength for ``selected`` debug PNG (0–1)."""

    debug_color_seed: int
    """PRNG seed for deterministic pseudo-colors in component viz."""

    # --- Skin / hand removal ---
    skin_removal_enabled: bool
    """When True, detect and remove hand/finger skin from the product mask."""

    skin_min_ratio: float
    """Minimum fraction of fg that must be skin before removal activates (0–1)."""

    skin_max_ratio: float
    """If skin exceeds this fraction, product may be skin-toned; skip removal."""

    skin_min_blob_pct: float
    """Minimum skin blob size as fraction of fg area to consider for removal."""

    skin_border_margin: float
    """Fraction of image dimension used as border proximity threshold."""

    skin_centroid_margin: float
    """Normalized distance from fg center beyond which a blob is 'peripheral'."""

    # --- Isolate v3: optional SAM semantic refinement (MobileSAM) ---
    semantic_refinement_enabled: bool
    """When True and checkpoint+deps exist, SAM may refine ambiguous multi-region cases."""

    semantic_sam_checkpoint: str | None
    """Path to MobileSAM ``mobile_sam.pt`` (or compatible) weights; unset disables loading."""

    semantic_sam_use_gpu: bool
    """Prefer CUDA for SAM when both this flag and hardware allow."""

    semantic_sam_points_per_side: int
    """Automatic mask grid density (lower → faster on CPU)."""

    semantic_sam_pred_iou_thresh: float
    """SAM ``predicted_iou`` filter (MobileSAM automatic masks)."""

    semantic_sam_stability_thresh: float
    """SAM stability score filter."""

    semantic_sam_min_mask_area: int
    """Drop tiny automatic masks (pixels)."""

    semantic_candidate_min_fg_ratio: float
    """Minimum fraction of rembg foreground a candidate must cover after intersection."""

    semantic_trigger_min_large_regions: int
    """Activate if at least this many ``large`` CC regions exist."""

    semantic_trigger_large_area_ratio: float
    """``large`` region: CC area ≥ this fraction of total foreground area."""

    semantic_trigger_v2_conf_below: float
    """Activate if best v2 confidence falls below this (ambiguous ranking)."""

    semantic_trigger_second_ratio_min: float
    """Activate if ``second_best / best`` v2 confidence ≥ this (close tie)."""

    semantic_trigger_border_min: float
    """Border-contact ratio threshold for top-two conflict trigger."""

    semantic_trigger_fg_area_ratio: float
    """Foreground / image area; used with multi-region bulky trigger."""

    semantic_trigger_single_border_contact: float
    """Single merged FG: activate when ``border_contact_ratio`` ≥ this."""

    semantic_trigger_single_solidity_max: float
    """Single merged FG: activate when ``solidity`` ≤ this (irregular / merged silhouette)."""

    semantic_trigger_single_elongation_min: float
    """Single merged FG: activate when elongation ≥ this."""

    semantic_trigger_single_fill_ratio_min: float
    """Single merged FG: activate when bbox fill (area / bbox) **below** this."""

    semantic_trigger_single_fg_ratio_min: float
    """Minimum ``fg_area / image_area`` before single-region geometry triggers apply."""

    semantic_trigger_single_complexity_min: float
    """Single merged FG: activate when contour complexity ≥ this (ragged / spread silhouette)."""

    semantic_weight_area: float
    semantic_weight_center: float
    semantic_weight_border: float
    semantic_weight_overlap: float
    """Weight for SAM vs rembg silhouette agreement (|mask∩fg|/|mask|)."""

    semantic_weight_sam_iou: float
    semantic_center_bias: float
    """Centrality falloff strength (separate knob from v2 ``center_bias``)."""

    semantic_border_gamma: float
    semantic_confidence_bias: float
    semantic_confidence_scale: float
    semantic_catastrophic_confidence: float
    """Absolute floor on calibrated confidence; only catastrophic model failure falls back."""

    semantic_min_select_confidence: float
    """Diagnostic / tuning reference only; ranking is comparative (see catastrophic floor)."""

    semantic_min_heuristic_agreement: float
    """Former hard-veto threshold — unused for rejection; kept for env / debug reference."""

    semantic_weight_heuristic_agree: float
    """Soft bonus when SAM mask overlaps heuristic CC keep (weak prior, not a veto)."""

    semantic_weight_compactness: float
    semantic_weight_fill_ratio: float
    semantic_weight_fragmentation: float
    """Penalty weight on mask fragmentation (disconnected spread)."""

    semantic_weight_merged_blob: float
    """Penalty when ``area_ratio`` exceeds merged-blob start (discourages full-FG swallow)."""

    semantic_merged_blob_start: float
    """``area_ratio`` above which merged-blob penalty ramps (0–1)."""

    semantic_comparative_margin_weight: float
    """Scales (best − second) logit margin into optional confidence lift."""

    semantic_comparative_sharpening: float
    """How much comparative margin lifts the winner's confidence (capped internally)."""

    semantic_debug_mask_blend: float
    """Blend strength for SAM debug overlay PNGs."""


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
        complexity_score_cap=_float_env("IMAGE_UTIL_ISOLATE_COMPLEXITY_CAP", 10.0),
        aspect_ratio_penalty_threshold=_float_env("IMAGE_UTIL_ISOLATE_ASPECT_RATIO_THRESH", 4.0),
        elongation_penalty=_float_env("IMAGE_UTIL_ISOLATE_ELONGATION_PENALTY", 0.65),
        math_epsilon=_float_env("IMAGE_UTIL_ISOLATE_MATH_EPS", 1e-6),
        v2_weight_relative_area=_float_env("IMAGE_UTIL_ISOLATE_V2_W_REL_AREA", 1.15),
        v2_weight_centrality=_float_env("IMAGE_UTIL_ISOLATE_V2_W_CENTRALITY", 1.35),
        v2_weight_complexity_norm=_float_env("IMAGE_UTIL_ISOLATE_V2_W_COMPLEXITY", 0.65),
        v2_weight_solidity=_float_env("IMAGE_UTIL_ISOLATE_V2_W_SOLIDITY", 0.55),
        v2_weight_border_contact=_float_env("IMAGE_UTIL_ISOLATE_V2_W_BORDER", 2.15),
        v2_border_contact_gamma=_float_env("IMAGE_UTIL_ISOLATE_V2_BORDER_GAMMA", 1.15),
        v2_weight_elongation_excess=_float_env("IMAGE_UTIL_ISOLATE_V2_W_ELONG", 0.35),
        v2_confidence_bias=_float_env("IMAGE_UTIL_ISOLATE_V2_CONF_BIAS", -0.15),
        v2_confidence_scale=_float_env("IMAGE_UTIL_ISOLATE_V2_CONF_SCALE", 1.0),
        v2_min_select_confidence=_float_env("IMAGE_UTIL_ISOLATE_V2_MIN_CONF", 0.06),
        v2_semantic_reject_relative=_float_env("IMAGE_UTIL_ISOLATE_V2_REJECT_REL", 0.38),
        v2_semantic_uncertain_relative=_float_env("IMAGE_UTIL_ISOLATE_V2_UNCERTAIN_REL", 0.72),
        solidity_floor=_float_env("IMAGE_UTIL_ISOLATE_SOLIDITY_FLOOR", 0.28),
        edge_blur_sigma=_float_env("IMAGE_UTIL_ISOLATE_EDGE_SIGMA", 0.85),
        rgb_zero_below_alpha=_int_env("IMAGE_UTIL_ISOLATE_RGB_ZERO_ALPHA", 8),
        rembg_model_name=os.getenv("IMAGE_UTIL_ISOLATE_REMBG_MODEL", "").strip() or None,
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
        debug_overlay_blend=_float_env("IMAGE_UTIL_ISOLATE_DEBUG_BLEND", 0.35),
        debug_color_seed=_int_env("IMAGE_UTIL_ISOLATE_DEBUG_COLOR_SEED", 42),
        skin_removal_enabled=_bool_env("IMAGE_UTIL_ISOLATE_SKIN_REMOVAL"),
        skin_min_ratio=_float_env("IMAGE_UTIL_ISOLATE_SKIN_MIN_RATIO", 0.05),
        skin_max_ratio=_float_env("IMAGE_UTIL_ISOLATE_SKIN_MAX_RATIO", 0.85),
        skin_min_blob_pct=_float_env("IMAGE_UTIL_ISOLATE_SKIN_MIN_BLOB_PCT", 0.02),
        skin_border_margin=_float_env("IMAGE_UTIL_ISOLATE_SKIN_BORDER_MARGIN", 0.05),
        skin_centroid_margin=_float_env("IMAGE_UTIL_ISOLATE_SKIN_CENTROID_MARGIN", 0.30),
        semantic_refinement_enabled=_bool_env("IMAGE_UTIL_ISOLATE_SEMANTIC"),
        semantic_sam_checkpoint=_str_env_optional("IMAGE_UTIL_ISOLATE_SAM_CHECKPOINT"),
        semantic_sam_use_gpu=_bool_env("IMAGE_UTIL_ISOLATE_SAM_GPU"),
        semantic_sam_points_per_side=_int_env("IMAGE_UTIL_ISOLATE_SAM_POINTS_PER_SIDE", 12),
        semantic_sam_pred_iou_thresh=_float_env("IMAGE_UTIL_ISOLATE_SAM_PRED_IOU_THRESH", 0.82),
        semantic_sam_stability_thresh=_float_env("IMAGE_UTIL_ISOLATE_SAM_STABILITY_THRESH", 0.9),
        semantic_sam_min_mask_area=_int_env("IMAGE_UTIL_ISOLATE_SAM_MIN_MASK_AREA", 180),
        semantic_candidate_min_fg_ratio=_float_env("IMAGE_UTIL_ISOLATE_SEM_CAND_MIN_FG", 0.06),
        semantic_trigger_min_large_regions=_int_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_LARGE_N", 2),
        semantic_trigger_large_area_ratio=_float_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_LARGE_RATIO", 0.12),
        semantic_trigger_v2_conf_below=_float_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_V2_BELOW", 0.68),
        semantic_trigger_second_ratio_min=_float_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_AMBIG", 0.82),
        semantic_trigger_border_min=_float_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_BORDER", 0.14),
        semantic_trigger_fg_area_ratio=_float_env("IMAGE_UTIL_ISOLATE_SEM_TRIG_FG_RATIO", 0.38),
        semantic_trigger_single_border_contact=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_BORDER", 0.13
        ),
        semantic_trigger_single_solidity_max=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_SOLIDITY_MAX", 0.44
        ),
        semantic_trigger_single_elongation_min=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_ELONG_MIN", 3.0
        ),
        semantic_trigger_single_fill_ratio_min=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_FILL_MIN", 0.42
        ),
        semantic_trigger_single_fg_ratio_min=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_FG_MIN", 0.10
        ),
        semantic_trigger_single_complexity_min=_float_env(
            "IMAGE_UTIL_ISOLATE_SEM_TRIG_SINGLE_COMPLEXITY_MIN", 7.0
        ),
        semantic_weight_area=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_AREA", 1.0),
        semantic_weight_center=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_CENTER", 1.2),
        semantic_weight_border=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_BORDER", 1.85),
        semantic_weight_overlap=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_OVERLAP", 0.9),
        semantic_weight_sam_iou=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_SAM_IOU", 0.75),
        semantic_center_bias=_float_env("IMAGE_UTIL_ISOLATE_SEM_CENTER_BIAS", 2.0),
        semantic_border_gamma=_float_env("IMAGE_UTIL_ISOLATE_SEM_BORDER_GAMMA", 1.12),
        semantic_confidence_bias=_float_env("IMAGE_UTIL_ISOLATE_SEM_CONF_BIAS", -0.12),
        semantic_confidence_scale=_float_env("IMAGE_UTIL_ISOLATE_SEM_CONF_SCALE", 1.0),
        semantic_catastrophic_confidence=_float_env("IMAGE_UTIL_ISOLATE_SEM_CATASTROPHIC", 0.07),
        semantic_min_select_confidence=_float_env("IMAGE_UTIL_ISOLATE_SEM_MIN_CONF", 0.22),
        semantic_min_heuristic_agreement=_float_env("IMAGE_UTIL_ISOLATE_SEM_MIN_AGREE", 0.28),
        semantic_weight_heuristic_agree=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_HEUR_AGREE", 0.42),
        semantic_weight_compactness=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_COMPACT", 0.55),
        semantic_weight_fill_ratio=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_FILL", 0.5),
        semantic_weight_fragmentation=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_FRAG", 0.95),
        semantic_weight_merged_blob=_float_env("IMAGE_UTIL_ISOLATE_SEM_W_MERGED", 1.35),
        semantic_merged_blob_start=_float_env("IMAGE_UTIL_ISOLATE_SEM_MERGED_START", 0.78),
        semantic_comparative_margin_weight=_float_env("IMAGE_UTIL_ISOLATE_SEM_MARGIN_W", 1.0),
        semantic_comparative_sharpening=_float_env("IMAGE_UTIL_ISOLATE_SEM_MARGIN_SHARP", 0.35),
        semantic_debug_mask_blend=_float_env("IMAGE_UTIL_ISOLATE_SEM_DEBUG_BLEND", 0.42),
    )
