"""Re-export grouping public surface."""

from __future__ import annotations

from .config import GroupingConfig, load_grouping_config
from .contracts import (
    GroupedCandidate,
    GroupingCandidateInput,
    GroupingFeatureVector,
    GroupingInput,
    GroupingMetadata,
    GroupingRankingSnapshot,
    GroupingResult,
    GroupRelationship,
)
from .processor import GroupingProcessor

__all__ = [
    "GroupedCandidate",
    "GroupingCandidateInput",
    "GroupingConfig",
    "GroupingFeatureVector",
    "GroupingInput",
    "GroupingMetadata",
    "GroupingProcessor",
    "GroupingRankingSnapshot",
    "GroupingResult",
    "GroupRelationship",
    "load_grouping_config",
]
