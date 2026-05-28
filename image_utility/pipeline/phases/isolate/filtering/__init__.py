"""Semantic proposal filtering (pre-ranking gate)."""

from __future__ import annotations

from .config import FilteringConfig, load_filtering_config, load_stop_after_aliases
from .contracts import (
    FilteringInput,
    FilteringMetadata,
    FilteringProposal,
    FilteringResult,
    FilteringScore,
    ScoredFilteringProposal,
)
from .processor import FilteringProcessor

__all__ = [
    "FilteringConfig",
    "FilteringInput",
    "FilteringMetadata",
    "FilteringProcessor",
    "FilteringProposal",
    "FilteringResult",
    "FilteringScore",
    "ScoredFilteringProposal",
    "load_filtering_config",
    "load_stop_after_aliases",
]
