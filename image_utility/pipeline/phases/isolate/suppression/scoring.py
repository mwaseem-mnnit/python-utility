"""Artifact / support-object likelihood — explainable, deterministic."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import SuppressionConfig
from .contracts import SuppressionGroupedRegionInput, SuppressionGroupScore

BoolMask = NDArray[np.bool_]


def _border_contact_ratio(mask: BoolMask, eps: float) -> float:
    h, w = mask.shape[:2]
    if h < 2 or w < 2 or not np.any(mask):
        return 0.0
    border = np.zeros_like(mask, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    inter = np.logical_and(mask, border.astype(bool))
    fg = int(np.count_nonzero(mask))
    return float(np.count_nonzero(inter) / max(fg, eps))


def _elongation_from_bbox(bw: int, bh: int, eps: float) -> float:
    bw, bh = max(bw, 1), max(bh, 1)
    return float(max(bw / bh, bh / bw))


def _secondary_blob_area_ratio(mask: BoolMask) -> float:
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    areas = sorted((float(cv2.contourArea(c)) for c in contours), reverse=True)
    if len(areas) < 2 or areas[0] < 1.0:
        return 0.0
    return float(areas[1] / areas[0])


def _contour_solidity(mask: BoolMask) -> float:
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 1.0:
        return 1.0
    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    if ha < 1.0:
        return 1.0
    return float(area / ha)


def _bbox_fill_ratio(region: SuppressionGroupedRegionInput, eps: float) -> float:
    gm = region.geometry_metadata
    bw = int(gm.get("bbox_w", 0))
    bh = int(gm.get("bbox_h", 0))
    bbox_a = float(max(bw * bh, 1))
    pix = float(int(gm.get("pixel_area", 0)))
    return float(pix / max(bbox_a, eps))


def score_grouped_region(
    region: SuppressionGroupedRegionInput,
    cfg: SuppressionConfig,
) -> SuppressionGroupScore:
    eps = cfg.math_epsilon
    mask = region.grouped_mask
    border = _border_contact_ratio(mask, eps)
    gm = region.geometry_metadata
    elong = _elongation_from_bbox(int(gm.get("bbox_w", 1)), int(gm.get("bbox_h", 1)), eps)
    elong_n = float(min(1.0, elong / max(cfg.elongation_critical, eps)))

    sec_ratio = _secondary_blob_area_ratio(mask)
    sec_n = float(min(1.0, sec_ratio / max(cfg.secondary_blob_critical, eps)))

    solidity = _contour_solidity(mask)
    thin_penalty = float(max(0.0, min(1.0, 1.0 - solidity)))

    bbox_fill = _bbox_fill_ratio(region, eps)
    sparse_penalty = float(max(0.0, min(1.0, 1.0 - min(1.0, bbox_fill / 0.12))))

    weak_sem = float(max(0.0, min(1.0, 1.0 - region.group_confidence)))

    internal = float(region.affinity_breakdown.get("internal_pair_affinity_mean", 1.0))
    loose_aff = float(max(0.0, min(1.0, (0.55 - internal) / 0.55)))

    # Ownership label penalty: non-product labels increase artifact likelihood.
    # ownership_label encoding: 1.0=product, 2.0=support_object, 3.0=packaging,
    # 4.0=environment, 5.0=uncertain.  Absent => 0.0 (no penalty).
    _OWN_PENALTY_MAP = {1.0: 0.0, 2.0: 1.0, 3.0: 0.15, 4.0: 0.85, 5.0: 0.40}
    own_label_id = float(region.affinity_breakdown.get("ownership_label", 0.0))
    own_label_penalty = float(_OWN_PENALTY_MAP.get(own_label_id, 0.0))

    weighted = (
        cfg.w_border * border
        + cfg.w_elongation * elong_n
        + cfg.w_secondary_blob * sec_n
        + cfg.w_thin_bridge * thin_penalty
        + cfg.w_bbox_fill_anomaly * sparse_penalty
        + cfg.w_weak_semantic * weak_sem
        + cfg.w_loose_affinity * loose_aff
        + cfg.w_ownership_label * own_label_penalty
    )
    wsum = (
        cfg.w_border
        + cfg.w_elongation
        + cfg.w_secondary_blob
        + cfg.w_thin_bridge
        + cfg.w_bbox_fill_anomaly
        + cfg.w_weak_semantic
        + cfg.w_loose_affinity
        + cfg.w_ownership_label
    )
    likelihood = weighted / max(wsum, eps)
    likelihood = float(max(0.0, min(1.0, likelihood)))

    bd = {
        "border_contact": border,
        "elongation_norm": elong_n,
        "secondary_blob_ratio": sec_ratio,
        "secondary_blob_norm": sec_n,
        "contour_solidity": solidity,
        "thin_bridge_penalty": thin_penalty,
        "bbox_fill_ratio": bbox_fill,
        "sparse_bbox_penalty": sparse_penalty,
        "weak_semantic_penalty": weak_sem,
        "loose_affinity_penalty": loose_aff,
        "ownership_label_penalty": own_label_penalty,
        "weighted_artifact_likelihood": likelihood,
    }
    return SuppressionGroupScore(
        group_id=region.group_id,
        artifact_likelihood=likelihood,
        suppression_breakdown=bd,
    )


def score_all_regions(
    regions: tuple[SuppressionGroupedRegionInput, ...],
    cfg: SuppressionConfig,
) -> tuple[SuppressionGroupScore, ...]:
    return tuple(score_grouped_region(r, cfg) for r in regions)
