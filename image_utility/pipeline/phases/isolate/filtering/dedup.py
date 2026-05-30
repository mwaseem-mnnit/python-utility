"""Pre-scoring mask deduplication — collapse near-identical SAM proposals.

SAM frequently generates many masks covering nearly the same pixels
(slightly shifted boundaries, overlapping sub-segments of the same region).
Deduplication reduces candidate volume before scoring so downstream ranking
and grouping operate on a cleaner, more discriminable set.

Deduplication is conservative:
- Only merges proposals with IoU above a high threshold (default 0.85)
- Always keeps the larger mask when merging a pair
- Operates in O(n²) pairwise but n is bounded by decomposition limits (~200)
"""

from __future__ import annotations

import logging

import numpy as np

from .config import FilteringConfig
from .contracts import FilteringProposal

LOGGER = logging.getLogger(__name__)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Fast boolean mask IoU."""
    inter = int(np.count_nonzero(np.logical_and(a, b)))
    union = int(np.count_nonzero(np.logical_or(a, b)))
    if union == 0:
        return 0.0
    return float(inter / union)


def deduplicate_proposals(
    proposals: tuple[FilteringProposal, ...],
    cfg: FilteringConfig,
) -> tuple[tuple[FilteringProposal, ...], int]:
    """
    Collapse near-identical masks.

    Returns:
        (deduplicated proposals, number of proposals removed)
    """
    if not cfg.dedup_enabled or len(proposals) < 2:
        return proposals, 0

    threshold = cfg.dedup_iou_threshold
    # Sort by area descending so larger masks survive when merging
    sorted_props = sorted(proposals, key=lambda p: p.area, reverse=True)

    kept: list[FilteringProposal] = []
    suppressed: set[int] = set()

    for i, pi in enumerate(sorted_props):
        if pi.candidate_id in suppressed:
            continue
        kept.append(pi)
        # Suppress all smaller proposals that are near-duplicates of this one
        for j in range(i + 1, len(sorted_props)):
            pj = sorted_props[j]
            if pj.candidate_id in suppressed:
                continue
            iou = _mask_iou(pi.mask, pj.mask)
            if iou >= threshold:
                suppressed.add(pj.candidate_id)

    removed = len(proposals) - len(kept)
    if removed > 0:
        LOGGER.info(
            "[filtering] dedup removed %d near-duplicate proposals (IoU>=%.2f), %d remain",
            removed,
            threshold,
            len(kept),
        )
    return tuple(kept), removed
