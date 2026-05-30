"""Filtering stage contracts — decomposition proposals enter only as these DTOs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class FilteringProposal:
    """Flattened decomposition semantic proposal — no decomposition imports downstream."""

    candidate_id: int
    mask: BoolMask
    source: str
    predicted_iou: float
    stability_score: float
    area: int


@dataclass(frozen=True)
class FilteringInput:
    proposals: tuple[FilteringProposal, ...]
    image_hw: tuple[int, int]


@dataclass(frozen=True)
class FilteringScore:
    validity_score: float
    rejection_likelihood: float
    heuristic_breakdown: dict[str, float]


@dataclass(frozen=True)
class ScoredFilteringProposal:
    proposal: FilteringProposal
    filtering_score: FilteringScore


@dataclass(frozen=True)
class FilteringMetadata:
    input_count: int
    accepted_count: int
    rejected_count: int
    all_rejected_fallback: bool
    dedup_removed_count: int = 0
    """Number of near-duplicate proposals collapsed before scoring."""
    post_dedup_count: int = 0
    """Proposals remaining after deduplication (equals input_count if dedup disabled)."""


@dataclass(frozen=True)
class FilteringResult:
    accepted: tuple[FilteringProposal, ...]
    rejected: tuple[FilteringProposal, ...]
    scored: tuple[ScoredFilteringProposal, ...]
    metadata: FilteringMetadata
