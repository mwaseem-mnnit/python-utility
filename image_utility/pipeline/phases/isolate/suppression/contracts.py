"""Suppression stage DTOs — semantic cleanup, not grouping or refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class SuppressionRankingSnapshot:
    """Optional upstream ranking summary (numeric only; no cross-stage types)."""

    candidate_count: int
    ambiguity_detected: bool
    top_confidence: float
    second_confidence: float
    confidence_separation: float


@dataclass(frozen=True)
class SuppressionGroupingSnapshot:
    """Optional upstream grouping summary."""

    group_count: int
    grouping_ambiguity: bool
    top_group_confidence: float
    second_group_confidence: float


@dataclass(frozen=True)
class SuppressionGroupedRegionInput:
    """One grouped region entering suppression (copied at isolate boundary)."""

    group_id: int
    member_candidate_ids: tuple[int, ...]
    grouped_mask: BoolMask
    group_confidence: float
    affinity_breakdown: dict[str, float]
    geometry_metadata: dict[str, float | int]


@dataclass(frozen=True)
class SuppressionInput:
    regions: tuple[SuppressionGroupedRegionInput, ...]
    image_hw: tuple[int, int]
    ranking_snapshot: SuppressionRankingSnapshot | None = None
    grouping_snapshot: SuppressionGroupingSnapshot | None = None


@dataclass(frozen=True)
class SuppressionGroupScore:
    """Artifact likelihood diagnostics for one grouped region."""

    group_id: int
    artifact_likelihood: float
    suppression_breakdown: dict[str, float]


@dataclass(frozen=True)
class SuppressedGroup:
    """A grouped region that survived cleanup (semantic mask unchanged per group)."""

    group_id: int
    member_candidate_ids: tuple[int, ...]
    surviving_mask: BoolMask
    removed_region_ids: tuple[int, ...]
    suppression_confidence: float
    suppression_breakdown: dict[str, float]
    geometry_metadata: dict[str, float | int]


@dataclass(frozen=True)
class SuppressionMetadata:
    analyzed_group_count: int
    removed_group_count: int
    surviving_group_count: int
    removed_group_ids: tuple[int, ...]
    global_suppression_confidence: float


@dataclass(frozen=True)
class SuppressionResult:
    surviving_groups: tuple[SuppressedGroup, ...]
    scores: tuple[SuppressionGroupScore, ...]
    combined_survivor_mask: BoolMask
    metadata: SuppressionMetadata
