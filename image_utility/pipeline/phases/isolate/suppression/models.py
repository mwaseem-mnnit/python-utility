"""Re-export suppression public API."""

from __future__ import annotations

from .config import SuppressionConfig, load_suppression_config
from .contracts import (
    SuppressedGroup,
    SuppressionGroupedRegionInput,
    SuppressionGroupingSnapshot,
    SuppressionGroupScore,
    SuppressionInput,
    SuppressionMetadata,
    SuppressionRankingSnapshot,
    SuppressionResult,
)
from .processor import SuppressionProcessor

__all__ = [
    "SuppressedGroup",
    "SuppressionConfig",
    "SuppressionGroupedRegionInput",
    "SuppressionGroupingSnapshot",
    "SuppressionGroupScore",
    "SuppressionInput",
    "SuppressionMetadata",
    "SuppressionProcessor",
    "SuppressionRankingSnapshot",
    "SuppressionResult",
    "load_suppression_config",
]
