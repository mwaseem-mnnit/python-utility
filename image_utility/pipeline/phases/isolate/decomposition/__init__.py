"""Semantic decomposition stage — high-recall proposals only."""

from __future__ import annotations

from .config import DecompositionConfig, load_decomposition_config
from .contracts import (
    ConnectedRegion,
    DecompositionMetadata,
    DecompositionResult,
    SemanticMaskCandidate,
)
from .processor import DecompositionProcessor

__all__ = [
    "ConnectedRegion",
    "DecompositionConfig",
    "DecompositionMetadata",
    "DecompositionProcessor",
    "DecompositionResult",
    "SemanticMaskCandidate",
    "load_decomposition_config",
]
