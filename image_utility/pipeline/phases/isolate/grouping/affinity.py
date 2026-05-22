"""Semantic affinity: explainable relatedness, not suppression scores."""

from __future__ import annotations

import math

from .config import GroupingConfig
from .contracts import GroupRelationship


def _sigmoid(x: float) -> float:
    if x >= 30.0:
        return 1.0
    if x <= -30.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def affinity_from_relationship(
    rel: GroupRelationship,
    cfg: GroupingConfig,
) -> tuple[float, dict[str, float]]:
    """
    Map relationship signals to [0, 1] affinity.
    Preserves overlapping high scores for multiple pairs (no winner-take-all).
    """
    centroid_proximity = 1.0 - min(1.0, rel.centroid_distance_norm * 1.25)
    edge_near = 1.0 - min(1.0, rel.edge_gap_norm)

    breakdown = {
        "overlap_ratio": rel.overlap_ratio,
        "mask_iou": rel.mask_iou,
        "centroid_proximity": float(centroid_proximity),
        "bbox_iou": rel.bbox_iou,
        "containment": rel.containment_smaller_in_larger,
        "edge_near": float(edge_near),
        "confidence_similarity": rel.confidence_similarity,
        "feature_consistency": rel.feature_consistency,
    }

    linear = (
        cfg.w_overlap * rel.overlap_ratio
        + cfg.w_mask_iou * rel.mask_iou
        + cfg.w_centroid * centroid_proximity
        + cfg.w_bbox_iou * rel.bbox_iou
        + cfg.w_containment * rel.containment_smaller_in_larger
        + cfg.w_edge * edge_near
        + cfg.w_confidence_sim * rel.confidence_similarity
        + cfg.w_feature_consistency * rel.feature_consistency
    )
    wsum = (
        cfg.w_overlap
        + cfg.w_mask_iou
        + cfg.w_centroid
        + cfg.w_bbox_iou
        + cfg.w_containment
        + cfg.w_edge
        + cfg.w_confidence_sim
        + cfg.w_feature_consistency
    )
    wsum = max(wsum, cfg.math_epsilon)
    fused = linear / wsum
    z = cfg.logit_scale * (fused - 0.5 + cfg.logit_bias)
    affinity = _sigmoid(z)
    affinity = float(max(cfg.confidence_floor, min(1.0, affinity)))
    breakdown["linear_fused"] = float(fused)
    breakdown["affinity"] = affinity
    return affinity, breakdown


def pair_affinity_table(
    relationships: tuple[GroupRelationship, ...],
    cfg: GroupingConfig,
) -> dict[tuple[int, int], tuple[float, dict[str, float]]]:
    table: dict[tuple[int, int], tuple[float, dict[str, float]]] = {}
    for rel in relationships:
        key = (min(rel.candidate_id_a, rel.candidate_id_b), max(rel.candidate_id_a, rel.candidate_id_b))
        table[key] = affinity_from_relationship(rel, cfg)
    return table
