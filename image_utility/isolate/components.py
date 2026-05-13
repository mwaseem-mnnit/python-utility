"""Connected-component analysis, rich features, and v2 confidence ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

UInt8 = NDArray[np.uint8]
Labels = NDArray[np.int32]

SEMANTIC_KEEP = "keep"
SEMANTIC_REJECT = "reject"
SEMANTIC_UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ComponentInfo:
    """Lightweight region summary (backward compatible)."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


@dataclass
class ComponentFeatures:
    """Per-component feature vector + v2 confidence and semantics."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    relative_area: float
    border_contact_ratio: float
    solidity: float
    elongation: float
    complexity: float
    confidence: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    semantic: str = SEMANTIC_UNCERTAIN

    def as_component_info(self) -> ComponentInfo:
        return ComponentInfo(
            label=self.label,
            area=self.area,
            bbox=self.bbox,
            centroid=self.centroid,
        )


def binary_foreground_mask(alpha: UInt8, cfg: IsolateConfig) -> UInt8:
    """Binary uint8 mask (0 / 255) where alpha indicates foreground."""
    return np.where(alpha > cfg.alpha_visibility_threshold, 255, 0).astype(np.uint8)


def analyze_connected_components(mask_255: UInt8) -> tuple[Labels, np.ndarray, np.ndarray]:
    """Returns ``(labels, stats, centroids)`` (label 0 = background)."""
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)
    _nlab, labels, stats, cents = cv2.connectedComponentsWithStats(mask_255)
    return labels, stats, cents


def _contour_complexity(mask_u8: UInt8, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    crop = mask_u8[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    a = float(cv2.contourArea(best))
    if a < 1.0:
        return 0.0
    return float(peri / math.sqrt(a))


def _solidity_in_bbox(mask_crop: UInt8) -> float:
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    if a < 1.0:
        return 1.0
    hull = cv2.convexHull(cnt)
    ha = float(cv2.contourArea(hull))
    return float(a / max(ha, 1.0))


def _border_contact_ratio(labels: Labels, label_id: int) -> float:
    m = labels == label_id
    area = int(np.count_nonzero(m))
    if area < 1:
        return 0.0
    edge = np.zeros_like(m, dtype=bool)
    edge[0, :] |= m[0, :]
    edge[-1, :] |= m[-1, :]
    edge[:, 0] |= m[:, 0]
    edge[:, -1] |= m[:, -1]
    return float(np.count_nonzero(np.logical_and(m, edge))) / float(area)


def _extract_features_for_label(
    label_id: int,
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    total_foreground_area: int,
    cfg: IsolateConfig,
) -> ComponentFeatures:
    x = int(stats[label_id, cv2.CC_STAT_LEFT])
    y = int(stats[label_id, cv2.CC_STAT_TOP])
    w = int(stats[label_id, cv2.CC_STAT_WIDTH])
    h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
    area = int(stats[label_id, cv2.CC_STAT_AREA])
    cx, cy = float(centroids[label_id, 0]), float(centroids[label_id, 1])
    bbox = (x, y, w, h)

    rel = float(area) / float(max(total_foreground_area, 1))
    border = _border_contact_ratio(labels, label_id)

    mask_bin = (labels == label_id).astype(np.uint8) * 255
    crop = mask_bin[y : y + h, x : x + w]
    solidity = _solidity_in_bbox(crop)

    elong = max(w, h) / max(min(w, h), 1)
    complexity = min(_contour_complexity(mask_bin, bbox), cfg.complexity_score_cap)

    return ComponentFeatures(
        label=label_id,
        area=area,
        bbox=bbox,
        centroid=(cx, cy),
        relative_area=rel,
        border_contact_ratio=border,
        solidity=solidity,
        elongation=elong,
        complexity=complexity,
    )


def _centrality_factor(cx: float, cy: float, ih: int, iw: int, cfg: IsolateConfig) -> float:
    cx0, cy0 = (iw - 1) * 0.5, (ih - 1) * 0.5
    diag = math.hypot(iw, ih)
    dist_norm = math.hypot(cx - cx0, cy - cy0) / (0.5 * diag + cfg.math_epsilon)
    return float(math.exp(-cfg.center_bias * dist_norm))


def _confidence_and_breakdown(
    feats: ComponentFeatures,
    image_hw: tuple[int, int],
    cfg: IsolateConfig,
) -> tuple[float, dict[str, float]]:
    ih, iw = image_hw
    cx, cy = feats.centroid

    centrality = _centrality_factor(cx, cy, ih, iw, cfg)
    comp_norm = min(feats.complexity / max(cfg.complexity_score_cap, 1e-6), 1.0)

    sol_clamped = max(feats.solidity, cfg.solidity_floor)
    solidity_excess = max(0.0, sol_clamped - cfg.solidity_floor)
    solidity_span = max(1.0 - cfg.solidity_floor, 1e-6)
    solidity_norm = min(solidity_excess / solidity_span, 1.0)

    border_term = feats.border_contact_ratio ** max(cfg.v2_border_contact_gamma, 1.0)
    elong_excess = max(0.0, feats.elongation - cfg.aspect_ratio_penalty_threshold)
    elong_term = math.sqrt(elong_excess)

    log_rel = math.log(feats.relative_area + 1e-6)

    z = cfg.v2_confidence_bias
    z += cfg.v2_weight_relative_area * log_rel
    z += cfg.v2_weight_centrality * centrality
    z += (cfg.v2_weight_complexity_norm + cfg.complexity_weight) * comp_norm
    z += cfg.v2_weight_solidity * solidity_norm
    z -= cfg.v2_weight_border_contact * border_term
    z -= cfg.v2_weight_elongation_excess * elong_term
    z = z * cfg.v2_confidence_scale
    conf = float(1.0 / (1.0 + math.exp(-z)))

    breakdown: dict[str, float] = {
        "logit": round(z, 4),
        "relative_area": round(feats.relative_area, 4),
        "centrality": round(centrality, 4),
        "complexity": round(feats.complexity, 4),
        "complexity_norm": round(comp_norm, 4),
        "solidity": round(feats.solidity, 4),
        "solidity_norm": round(solidity_norm, 4),
        "border_contact_ratio": round(feats.border_contact_ratio, 4),
        "border_term": round(border_term, 4),
        "elongation": round(feats.elongation, 4),
        "elong_excess": round(elong_excess, 4),
        "elong_term": round(elong_term, 4),
    }
    return conf, breakdown


def assign_component_semantics(
    ranked: list[ComponentFeatures],
    winner_label: int | None,
    cfg: IsolateConfig,
) -> None:
    """Label *keep* / *reject* / *uncertain* after a winner (if any) is chosen."""
    if not ranked:
        return
    winner_conf = 0.0
    if winner_label is not None:
        for f in ranked:
            if f.label == winner_label:
                winner_conf = f.confidence
                break
    ref = winner_conf if winner_label is not None else max(
        (f.confidence for f in ranked),
        default=0.0,
    )
    for f in ranked:
        if f.area < cfg.min_component_area:
            f.semantic = SEMANTIC_REJECT
            continue
        if winner_label is not None and f.label == winner_label:
            f.semantic = SEMANTIC_KEEP
            continue
        rel = f.confidence / max(ref, 1e-6)
        if rel < cfg.v2_semantic_reject_relative:
            f.semantic = SEMANTIC_REJECT
        elif rel < cfg.v2_semantic_uncertain_relative:
            f.semantic = SEMANTIC_UNCERTAIN
        else:
            f.semantic = SEMANTIC_UNCERTAIN


def rank_foreground_components(
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: IsolateConfig,
) -> list[ComponentFeatures]:
    """Build feature vectors and confidence; sort descending. Semantics finalized by selection."""
    ih, iw = labels.shape[:2]
    n = stats.shape[0]
    if n <= 1:
        return []

    total_fg = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))
    ranked: list[ComponentFeatures] = []
    for i in range(1, n):
        feats = _extract_features_for_label(i, labels, stats, centroids, total_fg, cfg)
        if feats.area < cfg.min_component_area:
            feats.confidence = 0.0
            feats.breakdown = {"rejected_by_area": 1.0}
            feats.semantic = SEMANTIC_REJECT
            ranked.append(feats)
            continue
        conf, bd = _confidence_and_breakdown(feats, (ih, iw), cfg)
        feats.confidence = conf
        feats.breakdown = bd
        feats.semantic = SEMANTIC_UNCERTAIN
        ranked.append(feats)

    ranked.sort(key=lambda f: f.confidence, reverse=True)
    return ranked


def select_best_component(
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: IsolateConfig,
) -> tuple[int | None, list[ComponentFeatures]]:
    """
    Rank by v2 confidence; return ``(label | None, ranked_features)``.

    Selection fails if no component meets ``v2_min_select_confidence``.
    """
    ranked = rank_foreground_components(labels, stats, centroids, cfg)
    viable = [f for f in ranked if f.area >= cfg.min_component_area and f.confidence > 0]
    if not viable:
        assign_component_semantics(ranked, None, cfg)
        return None, ranked
    best = viable[0]
    if best.confidence < cfg.v2_min_select_confidence:
        assign_component_semantics(ranked, None, cfg)
        return None, ranked
    assign_component_semantics(ranked, best.label, cfg)
    return best.label, ranked


def select_product_label(
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: IsolateConfig,
) -> tuple[int | None, list[ComponentInfo], dict[int, float]]:
    """Backward-compatible selection API (infos + raw confidence map)."""
    best, ranked = select_best_component(labels, stats, centroids, cfg)
    infos = [f.as_component_info() for f in ranked]
    scores = {f.label: round(float(f.confidence), 6) for f in ranked}
    return best, infos, scores


def apply_kept_label_to_alpha(alpha: UInt8, labels: Labels, keep_label: int) -> UInt8:
    """Zero alpha outside the kept connected-component region (preserve soft alpha inside)."""
    mask = labels == keep_label
    out = alpha.copy()
    out[~mask] = 0
    return out
