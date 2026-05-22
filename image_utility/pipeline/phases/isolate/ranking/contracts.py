"""Ranking stage DTOs — evidence only, not selection authority."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]
UInt8 = NDArray[np.uint8]


@dataclass(frozen=True)
class RankingMaskInput:
    """One decomposition proposal (copied at isolate boundary; avoids decomposition import)."""

    candidate_id: int
    mask: BoolMask
    source: str
    predicted_iou: float
    stability_score: float
    area: int


@dataclass(frozen=True)
class RankingInput:
    """Inputs required for ranking; built by isolate orchestration from decomposition output."""

    proposals: tuple[RankingMaskInput, ...]
    base_alpha: UInt8
    image_hw: tuple[int, int]


@dataclass(frozen=True)
class FeatureVector:
    """Explainable geometry / topology descriptor for one proposal."""

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
class RankedCandidate:
    """Scored proposal — multiple may remain plausible (ambiguity preserved)."""

    candidate_id: int
    source: str
    mask: BoolMask
    confidence: float
    features: FeatureVector
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RankingMetadata:
    """Run summary for tuning and ambiguity review."""

    candidate_count: int
    ambiguity_detected: bool
    top_confidence: float
    second_confidence: float
    confidence_separation: float


@dataclass(frozen=True)
class RankingResult:
    """Ordered semantic evidence; `ranked` sorted by confidence descending."""

    ranked: tuple[RankedCandidate, ...]
    metadata: RankingMetadata
