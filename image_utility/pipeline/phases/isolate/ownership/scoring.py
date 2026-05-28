"""Ownership scoring — four competing likelihoods, deterministic weighted logits."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .config import OwnershipConfig
from .contracts import (
    OwnershipFeatures,
    OwnershipGroupedRegionInput,
    OwnershipLabel,
    OwnershipScore,
)
if TYPE_CHECKING:
    pass
def _sigmoid(x: float) -> float:
    import math
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))
def _wsum(values: list[float], weights: list[float], eps: float) -> float:
    """Weighted mean, clamped to [0, 1]."""
    w_total = sum(weights)
    if w_total < eps:
        return 0.0
    raw = sum(v * w for v, w in zip(values, weights)) / w_total
    return float(np.clip(raw, 0.0, 1.0))
def _assign_label(
    product: float,
    support: float,
    packaging: float,
    env: float,
    cfg: OwnershipConfig,
) -> OwnershipLabel:
    """Greedy label assignment: highest likelihood wins if above its threshold."""
    candidates: list[tuple[float, float, OwnershipLabel]] = [
        (support, cfg.support_label_threshold, "support_object"),
        (env, cfg.environment_label_threshold, "environment"),
        (packaging, cfg.packaging_label_threshold, "packaging"),
        (product, cfg.product_label_threshold, "product"),
    ]
    # Only consider candidates that clear their own threshold
    cleared = [(score, label) for score, thresh, label in candidates if score >= thresh]
    if not cleared:
        return "uncertain"
    return max(cleared, key=lambda t: t[0])[1]
def score_group(
    region: OwnershipGroupedRegionInput,
    features: OwnershipFeatures,
    cfg: OwnershipConfig,
    anchor_group_id: int | None,
) -> OwnershipScore:
    """Compute four ownership likelihoods for one group."""
    eps = cfg.math_epsilon
    f = features
    is_anchor = region.group_id == anchor_group_id
    # ------------------------------------------------------------------
    # Product likelihood
    # ------------------------------------------------------------------
    centre_score = float(np.clip(1.0 - f.center_distance_norm * 2.5, 0.0, 1.0))
    conf_score = float(np.clip(region.group_confidence, 0.0, 1.0))
    solidity_score = float(np.clip(f.solidity, 0.0, 1.0))
    # Low elongation = compact = product-like (elongation 1.0 => best)
    compactness_score = float(np.clip(1.0 - (f.elongation - 1.0) / max(cfg.support_min_elongation, eps), 0.0, 1.0))
    product_raw = _wsum(
        [centre_score, conf_score, solidity_score, compactness_score],
        [cfg.w_product_centre, cfg.w_product_confidence, cfg.w_product_solidity, cfg.w_product_compactness],
        eps,
    )
    # Anchor group gets a direct confidence boost
    if is_anchor:
        product_raw = float(np.clip(product_raw + 0.12, 0.0, 1.0))
    # ------------------------------------------------------------------
    # Support-object likelihood (hand/finger/stand/holder)
    # ------------------------------------------------------------------
    elong_score = float(np.clip(
        (f.elongation - 1.0) / max(cfg.support_min_elongation - 1.0, eps), 0.0, 1.0
    ))
    low_solid_score = float(np.clip(
        (cfg.support_max_solidity - f.solidity) / max(cfg.support_max_solidity, eps), 0.0, 1.0
    ))
    complex_score = float(np.clip(
        (f.contour_complexity - 1.0) / max(cfg.support_min_contour_complexity - 1.0, eps), 0.0, 1.0
    ))
    sec_blob_score = float(np.clip(
        f.secondary_blob_ratio / max(cfg.support_min_secondary_blob_ratio, eps), 0.0, 1.0
    ))
    bridge_score = float(np.clip(f.thin_bridge_score, 0.0, 1.0))
    finger_score = float(np.clip(
        f.finger_like_ratio / max(cfg.support_min_finger_like_ratio, eps), 0.0, 1.0
    ))
    periph_score = float(np.clip(f.center_distance_norm * 2.0, 0.0, 1.0))
    support_raw = _wsum(
        [elong_score, low_solid_score, complex_score, sec_blob_score,
         bridge_score, finger_score, periph_score],
        [cfg.w_support_elongation, cfg.w_support_low_solidity, cfg.w_support_complexity,
         cfg.w_support_secondary_blob, cfg.w_support_thin_bridge, cfg.w_support_finger_like,
         cfg.w_support_periphery],
        eps,
    )
    # Anchor group cannot be a support object
    if is_anchor:
        support_raw = float(np.clip(support_raw * 0.25, 0.0, 1.0))
    # ------------------------------------------------------------------
    # Environment likelihood (table/background slab)
    # ------------------------------------------------------------------
    area_env = float(np.clip(f.relative_area / max(cfg.env_max_image_ratio, eps), 0.0, 1.0))
    border_env = float(np.clip(f.border_contact_ratio / max(cfg.env_min_border_contact, eps), 0.0, 1.0))
    env_raw = _wsum([area_env, border_env], [cfg.w_env_area, cfg.w_env_border], eps)
    # ------------------------------------------------------------------
    # Packaging likelihood (box/wrapping adjacent to product)
    # ------------------------------------------------------------------
    fill_pkg = float(np.clip(f.bbox_fill_ratio / max(cfg.pkg_min_bbox_fill, eps), 0.0, 1.0))
    # Packaging is compact; penalise for elongation
    compact_pkg = float(np.clip(
        1.0 - (f.elongation - 1.0) / max(cfg.pkg_max_elongation - 1.0, eps), 0.0, 1.0
    ))
    pkg_raw = _wsum([fill_pkg, compact_pkg], [cfg.w_pkg_bbox_fill, cfg.w_pkg_compactness], eps)
    # Only plausible if not already strongly product or support
    pkg_raw = float(pkg_raw * (1.0 - max(product_raw, support_raw) * 0.6))
    # ------------------------------------------------------------------
    # Label assignment
    # ------------------------------------------------------------------
    label = _assign_label(product_raw, support_raw, pkg_raw, env_raw, cfg)
    breakdown = {
        "product_likelihood": round(product_raw, 5),
        "support_likelihood": round(support_raw, 5),
        "packaging_likelihood": round(pkg_raw, 5),
        "environment_likelihood": round(env_raw, 5),
        # product signals
        "centre_score": round(centre_score, 5),
        "conf_score": round(conf_score, 5),
        "solidity_score": round(solidity_score, 5),
        "compactness_score": round(compactness_score, 5),
        "is_anchor": 1.0 if is_anchor else 0.0,
        # support signals
        "elongation_score": round(elong_score, 5),
        "low_solidity_score": round(low_solid_score, 5),
        "complexity_score": round(complex_score, 5),
        "secondary_blob_score": round(sec_blob_score, 5),
        "bridge_score": round(bridge_score, 5),
        "finger_score": round(finger_score, 5),
        "periphery_score": round(periph_score, 5),
    }
    return OwnershipScore(
        group_id=region.group_id,
        product_likelihood=product_raw,
        support_likelihood=support_raw,
        packaging_likelihood=pkg_raw,
        environment_likelihood=env_raw,
        assigned_label=label,
        score_breakdown=breakdown,
        features=features,
    )
def score_all_groups(
    regions: tuple[OwnershipGroupedRegionInput, ...],
    features: tuple[OwnershipFeatures, ...],
    cfg: OwnershipConfig,
    anchor_group_id: int | None,
) -> tuple[OwnershipScore, ...]:
    return tuple(
        score_group(r, f, cfg, anchor_group_id)
        for r, f in zip(regions, features)
    )
