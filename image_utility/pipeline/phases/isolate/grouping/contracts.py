"""Grouping stage DTOs — relationship analysis only, not suppression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]
UInt8 = NDArray[np.uint8]


@dataclass(frozen=True)
class GroupingFeatureVector:
    """Geometry / topology descriptor (mirrors ranking features; stage-local copy)."""

    area: int
    relative_area: float
    centroid_xy: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int]
    bbox_fill_ratio: float
    solidity: float
    elongation: float
    border_contact_ratio: float
    contour_complexity: float
    occupancy_ratio: float
    center_distance_norm: float
    sam_predicted_iou: float
    sam_stability: float
    overlap_rembg_fg: float


@dataclass(frozen=True)
class GroupingCandidateInput:
    """One ranked semantic candidate entering grouping (no cross-stage types)."""

    candidate_id: int
    source: str
    mask: BoolMask
    confidence: float
    features: GroupingFeatureVector
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupingRankingSnapshot:
    """Optional summary of the upstream ranking pass (for metadata continuity)."""

    candidate_count: int
    ambiguity_detected: bool
    top_confidence: float
    second_confidence: float
    confidence_separation: float


@dataclass(frozen=True)
class GroupingInput:
    proposals: tuple[GroupingCandidateInput, ...]
    base_alpha: UInt8
    image_hw: tuple[int, int]
    ranking_snapshot: GroupingRankingSnapshot | None = None


@dataclass(frozen=True)
class GroupRelationship:
    """Explainable pairwise geometry + soft semantic agreement."""

    candidate_id_a: int
    candidate_id_b: int
    overlap_ratio: float
    mask_iou: float
    centroid_distance_norm: float
    bbox_iou: float
    containment_smaller_in_larger: float
    edge_gap_norm: float
    confidence_similarity: float
    feature_consistency: float


@dataclass(frozen=True)
class GroupedCandidate:
    """Merged evidence for a product / multi-part structure (may overlap others)."""

    group_id: int
    member_candidate_ids: tuple[int, ...]
    grouped_mask: BoolMask
    group_confidence: float
    affinity_breakdown: dict[str, float]
    relationship_pairs: tuple[tuple[int, int, float], ...]
    geometry_metadata: dict[str, float | int]


@dataclass(frozen=True)
class GroupingMetadata:
    group_count: int
    candidate_count: int
    multi_member_group_count: int
    top_group_confidence: float
    second_group_confidence: float
    grouping_ambiguity: bool


@dataclass(frozen=True)
class GroupingResult:
    groups: tuple[GroupedCandidate, ...]
    relationships: tuple[GroupRelationship, ...]
    pair_affinities: tuple[tuple[int, int, float], ...]
    metadata: GroupingMetadata
