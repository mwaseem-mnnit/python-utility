"""Score semantic (SAM) regions for product-vs-artifact ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import IsolateConfig
from .masks import PreparedMask

BoolMask = NDArray[np.bool_]


def _border_contact_ratio(mask: BoolMask) -> float:
    area = int(np.count_nonzero(mask))
    if area < 1:
        return 0.0
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] |= mask[0, :]
    edge[-1, :] |= mask[-1, :]
    edge[:, 0] |= mask[:, 0]
    edge[:, -1] |= mask[:, -1]
    return float(np.count_nonzero(np.logical_and(mask, edge))) / float(area)


def _centrality(mask: BoolMask, ih: int, iw: int, cfg: IsolateConfig) -> float:
    m = mask.astype(np.uint8)
    mxs = m.sum(axis=1)
    mys = m.sum(axis=0)
    if mxs.max() < 1 or mys.max() < 1:
        return 0.0
    ys = np.arange(ih, dtype=np.float64)
    xs = np.arange(iw, dtype=np.float64)
    cy = float((mxs * ys).sum() / mxs.sum())
    cx = float((mys * xs).sum() / mys.sum())
    cx0, cy0 = (iw - 1) * 0.5, (ih - 1) * 0.5
    diag = math.hypot(iw, ih)
    dist_norm = math.hypot(cx - cx0, cy - cy0) / (0.5 * diag + cfg.math_epsilon)
    return float(math.exp(-cfg.semantic_center_bias * dist_norm))


def _compactness(mask: BoolMask) -> float:
    u8 = (mask.astype(np.uint8) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    p = float(cv2.arcLength(cnt, True))
    if p < 1e-3:
        return 0.0
    return float(min(4.0 * math.pi * a / (p * p), 1.0))


def _bbox_fill_ratio(mask: BoolMask) -> float:
    ys, xs = np.where(mask)
    if ys.size < 1:
        return 0.0
    area = float(ys.size)
    h = float(ys.max() - ys.min() + 1)
    w = float(xs.max() - xs.min() + 1)
    box = max(h * w, 1.0)
    return float(min(area / box, 1.0))


def _fragmentation(mask: BoolMask) -> float:
    """1 − (largest CC area / mask area); high ⇒ scattered / disconnected."""
    u8 = mask.astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 1:
        return 0.0
    areas = [int(st[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    total = sum(areas)
    largest = max(areas) if areas else 0
    return float(1.0 - (largest / max(total, 1)))


def _heuristic_agreement(pm: PreparedMask, heuristic_mask: BoolMask | None) -> float:
    if heuristic_mask is None:
        return 0.0
    ha = int(np.count_nonzero(heuristic_mask))
    if ha < 1:
        return 0.0
    inter = int(np.count_nonzero(np.logical_and(pm.mask, heuristic_mask)))
    return float(inter) / float(ha)


def _semantic_dominance_index(
    compactness: float,
    fill: float,
    border: float,
    fragmentation: float,
) -> float:
    """Lightweight 0–1 summary for logging / debug (not a second gate)."""
    b = max(0.0, min(1.0, 1.0 - border))
    f = max(0.0, min(1.0, 1.0 - fragmentation))
    return float(
        max(0.0, min(1.0, 0.28 * compactness + 0.28 * fill + 0.22 * b + 0.22 * f))
    )


@dataclass
class SemanticRegionScore:
    region_id: int
    mask: BoolMask
    confidence: float
    breakdown: dict[str, float] = field(default_factory=dict)


def rank_semantic_regions(
    candidates: list[PreparedMask],
    fg_bool: BoolMask,
    image_hw: tuple[int, int],
    cfg: IsolateConfig,
    heuristic_mask: BoolMask | None = None,
) -> list[SemanticRegionScore]:
    ih, iw = image_hw
    fg_area = max(int(np.count_nonzero(fg_bool)), 1)

    scored: list[SemanticRegionScore] = []
    confidences_raw: list[float] = []
    for idx, pm in enumerate(candidates):
        area = int(np.count_nonzero(pm.mask))
        area_ratio = float(area) / float(fg_area)
        border = _border_contact_ratio(pm.mask)
        border_term = border ** max(cfg.semantic_border_gamma, 1.0)
        centrality = _centrality(pm.mask, ih, iw, cfg)
        sam_term = max(pm.pred_iou, pm.stability * 0.5)
        compact = _compactness(pm.mask)
        fill = _bbox_fill_ratio(pm.mask)
        frag = _fragmentation(pm.mask)
        agree_h = _heuristic_agreement(pm, heuristic_mask)
        merged_excess = max(0.0, area_ratio - cfg.semantic_merged_blob_start)

        z = cfg.semantic_confidence_bias
        z += cfg.semantic_weight_area * math.log(area_ratio + cfg.math_epsilon)
        z += cfg.semantic_weight_center * centrality
        z -= cfg.semantic_weight_border * border_term
        z += cfg.semantic_weight_overlap * pm.overlap_rembg
        z += cfg.semantic_weight_sam_iou * sam_term
        z += cfg.semantic_weight_heuristic_agree * agree_h
        z += cfg.semantic_weight_compactness * compact
        z += cfg.semantic_weight_fill_ratio * fill
        z -= cfg.semantic_weight_fragmentation * frag
        z -= cfg.semantic_weight_merged_blob * (merged_excess**2)

        z_pre = z * cfg.semantic_confidence_scale
        conf = float(1.0 / (1.0 + math.exp(-z_pre)))
        confidences_raw.append(conf)

        dom = _semantic_dominance_index(compact, fill, border, frag)

        breakdown = {
            "logit": round(z_pre, 4),
            "area_ratio": round(area_ratio, 4),
            "border_contact": round(border, 4),
            "centrality": round(centrality, 4),
            "overlap_rembg": round(pm.overlap_rembg, 4),
            "pred_iou": round(pm.pred_iou, 4),
            "stability": round(pm.stability, 4),
            "heuristic_agreement": round(agree_h, 4),
            "compactness": round(compact, 4),
            "fill_ratio": round(fill, 4),
            "fragmentation": round(frag, 4),
            "merged_excess": round(merged_excess, 4),
            "semantic_dominance": round(dom, 4),
        }
        scored.append(
            SemanticRegionScore(region_id=idx, mask=pm.mask, confidence=conf, breakdown=breakdown)
        )

    scored.sort(key=lambda s: s.confidence, reverse=True)

    # Comparative margin: lift winner slightly when it clearly leads (soft, not a hard gate).
    if len(scored) >= 2:
        best_z = scored[0].breakdown["logit"]
        second_z = scored[1].breakdown["logit"]
        margin = max(0.0, (best_z - second_z) * cfg.semantic_comparative_margin_weight)
        if margin > 0 and cfg.semantic_comparative_sharpening > 0:
            delta = min(0.12, margin * cfg.semantic_comparative_sharpening)
            top = scored[0]
            c = top.confidence
            top.confidence = float(max(0.0, min(1.0, c + delta)))
            top.breakdown["comparative_lift"] = round(delta, 4)

    return scored
