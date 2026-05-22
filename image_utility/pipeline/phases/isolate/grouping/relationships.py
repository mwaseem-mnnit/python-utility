"""Pairwise relationship analysis (stage-local, deterministic)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .contracts import GroupRelationship, GroupingCandidateInput

BoolMask = NDArray[np.bool_]


def _bbox_intersection_xywh(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _bbox_area_xywh(box: tuple[int, int, int, int]) -> int:
    return int(max(0, box[2]) * max(0, box[3]))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter = _bbox_intersection_xywh(a, b)
    if inter is None:
        return 0.0
    ia = inter[2] * inter[3]
    ua = _bbox_area_xywh(a) + _bbox_area_xywh(b) - ia
    if ua <= 0:
        return 0.0
    return float(ia / ua)


def _edge_gap_norm(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    diagonal: float,
    eps: float,
) -> float:
    """0 when bboxes overlap or touch; approaches 1 when far apart (clamped)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_x2, a_y2 = ax + aw, ay + ah
    b_x2, b_y2 = bx + bw, by + bh
    dx = max(0, max(ax, bx) - min(a_x2, b_x2))
    dy = max(0, max(ay, by) - min(a_y2, b_y2))
    gap = math.hypot(float(dx), float(dy))
    return float(min(1.0, gap / max(diagonal, eps)))


def _feature_consistency(fi: GroupingCandidateInput, fj: GroupingCandidateInput, eps: float) -> float:
    """Soft agreement on ranking-derived cues (not suppression)."""
    af = fi.features
    bf = fj.features
    # Normalized L1 distance on a small stable subset
    keys = (
        af.overlap_rembg_fg,
        bf.overlap_rembg_fg,
        af.relative_area,
        bf.relative_area,
        af.solidity,
        bf.solidity,
        af.sam_predicted_iou,
        bf.sam_predicted_iou,
        float(af.area),
        float(bf.area),
    )
    # scale area by hypothetic max — use sum of areas as scale
    scale = max(float(fi.features.area + fj.features.area), eps)
    d1 = abs(keys[0] - keys[1])
    d2 = abs(keys[2] - keys[3])
    d3 = abs(keys[4] - keys[5])
    d4 = abs(keys[6] - keys[7])
    d5 = abs(keys[8] - keys[9]) / scale
    dist = (d1 + d2 + d3 + d4 + min(1.0, d5)) / 5.0
    return float(max(0.0, 1.0 - dist))


def pairwise_relationship(
    ci: GroupingCandidateInput,
    cj: GroupingCandidateInput,
    *,
    image_hw: tuple[int, int],
    eps: float,
) -> GroupRelationship:
    h, w = image_hw
    diagonal = float(math.hypot(max(w, 1), max(h, 1)))

    mi, mj = ci.mask, cj.mask
    inter = np.logical_and(mi, mj)
    union = np.logical_or(mi, mj)
    inter_ct = int(np.count_nonzero(inter))
    union_ct = int(np.count_nonzero(union))
    ai, aj = int(ci.features.area), int(cj.features.area)
    denom_min = max(min(ai, aj), 1)
    overlap_ratio = float(min(1.0, inter_ct / denom_min))
    mask_iou = float(inter_ct / max(union_ct, 1))

    cxi, cyi = ci.features.centroid_xy
    cxj, cyj = cj.features.centroid_xy
    centroid_distance_norm = float(
        math.hypot(cxi - cxj, cyi - cyj) / max(diagonal, eps)
    )

    bbox_iou = _bbox_iou(ci.features.bbox_xywh, cj.features.bbox_xywh)

    smaller, larger = (ai, aj) if ai <= aj else (aj, ai)
    if smaller <= 0:
        containment = 0.0
    else:
        containment = float(min(1.0, inter_ct / smaller))

    edge_gap_norm = _edge_gap_norm(
        ci.features.bbox_xywh, cj.features.bbox_xywh, diagonal, eps
    )

    confidence_similarity = float(
        max(0.0, 1.0 - abs(ci.confidence - cj.confidence))
    )
    feature_consistency = _feature_consistency(ci, cj, eps)

    return GroupRelationship(
        candidate_id_a=ci.candidate_id,
        candidate_id_b=cj.candidate_id,
        overlap_ratio=overlap_ratio,
        mask_iou=mask_iou,
        centroid_distance_norm=centroid_distance_norm,
        bbox_iou=bbox_iou,
        containment_smaller_in_larger=containment,
        edge_gap_norm=edge_gap_norm,
        confidence_similarity=confidence_similarity,
        feature_consistency=feature_consistency,
    )


def build_pairwise_relationships(
    candidates: tuple[GroupingCandidateInput, ...],
    *,
    image_hw: tuple[int, int],
    eps: float,
) -> tuple[GroupRelationship, ...]:
    n = len(candidates)
    out: list[GroupRelationship] = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(
                pairwise_relationship(candidates[i], candidates[j], image_hw=image_hw, eps=eps)
            )
    return tuple(out)
