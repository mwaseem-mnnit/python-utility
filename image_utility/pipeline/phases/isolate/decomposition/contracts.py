"""Typed decomposition outputs (semantic proposals, topology metadata)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

UInt8 = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]
Labels = NDArray[np.int32]


@dataclass(frozen=True)
class ConnectedRegion:
    """Single connected-component region (decomposition metadata only; not a winner)."""

    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    contour_point_count: int
    solidity: float


@dataclass(frozen=True)
class SemanticMaskCandidate:
    """High-recall semantic proposal; overlapping / ambiguous masks allowed."""

    candidate_id: int
    mask: BoolMask
    source: str
    predicted_iou: float = 0.0
    stability_score: float = 0.0
    area: int = 0


@dataclass(frozen=True)
class DecompositionMetadata:
    """Tuning-friendly run summary (counts and provenance; not ranking scores)."""

    rembg_model: str | None
    morph_pre_close: int
    sam_enabled: bool
    sam_raw_mask_count: int
    normalized_candidate_count: int
    connected_region_count: int
    alpha_candidate_count: int
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DecompositionResult:
    """Full decomposition stage output — proposals only, no final selection."""

    base_rgba: UInt8
    base_alpha: UInt8
    cc_labels: Labels
    cc_stats: np.ndarray
    cc_centroids: np.ndarray
    connected_regions: tuple[ConnectedRegion, ...]
    semantic_candidates: tuple[SemanticMaskCandidate, ...]
    alpha_candidates: tuple[UInt8, ...]
    metadata: DecompositionMetadata
