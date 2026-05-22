"""Semantic confidence and explainable breakdown (soft scoring, no suppression)."""

from __future__ import annotations

import math

from .config import RankingConfig
from .contracts import FeatureVector, RankedCandidate, RankingMaskInput


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))


def score_candidate(
    prop: RankingMaskInput,
    feats: FeatureVector,
    cfg: RankingConfig,
) -> RankedCandidate:
    """
    Map features to a calibrated confidence and per-term breakdown.
    All candidates receive a score — nothing is hard-rejected here.
    """
    centrality = math.exp(-cfg.center_bias * feats.center_distance_norm)
    border_term = feats.border_contact_ratio ** max(cfg.border_gamma, 1.0)
    sol_eff = max(feats.solidity, cfg.solidity_floor)
    solidity_excess = max(0.0, sol_eff - cfg.solidity_floor)
    solidity_span = max(1.0 - cfg.solidity_floor, cfg.math_epsilon)
    solidity_norm = min(solidity_excess / solidity_span, 1.0)

    elong_excess = max(0.0, feats.elongation - cfg.elongation_thresh)
    elong_term = math.sqrt(elong_excess)

    comp_norm = min(feats.contour_complexity / max(cfg.complexity_cap, cfg.math_epsilon), 1.0)
    log_rel = math.log(feats.relative_area + cfg.math_epsilon)
    sam_term = feats.sam_predicted_iou + 0.5 * feats.sam_stability

    z = cfg.logit_bias
    z += cfg.area_log_weight * log_rel
    z += cfg.center_score_weight * centrality
    z -= cfg.border_penalty_weight * border_term
    z += cfg.solidity_weight * solidity_norm
    z -= cfg.elongation_penalty_weight * elong_term
    z += cfg.complexity_weight * comp_norm
    z += cfg.sam_iou_weight * sam_term
    z += cfg.overlap_rembg_weight * feats.overlap_rembg_fg
    z *= cfg.logit_scale

    conf = _sigmoid(z)
    conf = max(conf, cfg.confidence_floor)

    breakdown = {
        "logit": round(z, 4),
        "area_score": round(log_rel, 4),
        "center_score": round(centrality, 4),
        "border_penalty": round(border_term, 4),
        "solidity_score": round(solidity_norm, 4),
        "elongation_penalty": round(elong_term, 4),
        "complexity_score": round(comp_norm, 4),
        "sam_evidence": round(sam_term, 4),
        "overlap_rembg": round(feats.overlap_rembg_fg, 4),
        "confidence": round(conf, 4),
    }
    return RankedCandidate(
        candidate_id=prop.candidate_id,
        source=prop.source,
        mask=prop.mask,
        confidence=conf,
        features=feats,
        score_breakdown=breakdown,
    )
