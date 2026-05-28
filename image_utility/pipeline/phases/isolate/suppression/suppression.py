"""Apply conservative suppression decisions (preserve at least one product structure)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import SuppressionConfig
from .contracts import (
    SuppressedGroup,
    SuppressionGroupedRegionInput,
    SuppressionGroupScore,
)

BoolMask = NDArray[np.bool_]


def suppress_groups(
    regions: tuple[SuppressionGroupedRegionInput, ...],
    scores: tuple[SuppressionGroupScore, ...],
    cfg: SuppressionConfig,
) -> tuple[tuple[SuppressedGroup, ...], tuple[int, ...], BoolMask]:
    """
    Drop groups whose artifact likelihood exceeds threshold, except we always retain
    the strongest survival signal (confidence × low artifact likelihood).
    """
    by_id = {r.group_id: r for r in regions}
    score_by_id = {s.group_id: s for s in scores}

    protect_gid = max(
        regions,
        key=lambda r: r.group_confidence * (1.0 - score_by_id[r.group_id].artifact_likelihood),
    ).group_id

    survivors_out: list[SuppressedGroup] = []
    removed: list[int] = []

    for r in sorted(regions, key=lambda x: x.group_id):
        s = score_by_id[r.group_id]
        remove = (
            s.artifact_likelihood >= cfg.removal_likelihood_threshold
            and r.group_id != protect_gid
        )
        conf = float(
            max(
                cfg.confidence_floor,
                min(1.0, (1.0 - s.artifact_likelihood) * r.group_confidence),
            )
        )

        bd = dict(s.suppression_breakdown)
        bd["keep_decision"] = 0.0 if remove else 1.0
        bd["protected_group"] = 1.0 if r.group_id == protect_gid else 0.0

        if remove:
            removed.append(r.group_id)
            continue

        survivors_out.append(
            SuppressedGroup(
                group_id=r.group_id,
                member_candidate_ids=r.member_candidate_ids,
                surviving_mask=np.ascontiguousarray(r.grouped_mask.copy()),
                removed_region_ids=(),
                suppression_confidence=conf,
                suppression_breakdown=bd,
                geometry_metadata=dict(r.geometry_metadata),
            )
        )

    rem_set = set(removed)

    if not survivors_out:
        rp = by_id[protect_gid]
        s = score_by_id[protect_gid]
        rem_set.discard(protect_gid)
        survivors_out = [
            SuppressedGroup(
                group_id=rp.group_id,
                member_candidate_ids=rp.member_candidate_ids,
                surviving_mask=np.ascontiguousarray(rp.grouped_mask.copy()),
                removed_region_ids=(),
                suppression_confidence=float(
                    max(cfg.confidence_floor, (1.0 - s.artifact_likelihood) * rp.group_confidence)
                ),
                suppression_breakdown=dict(s.suppression_breakdown),
                geometry_metadata=dict(rp.geometry_metadata),
            )
        ]
        removed = [x for x in removed if x != protect_gid]
        rem_set = set(removed)

    survivors = tuple(
        SuppressedGroup(
            group_id=g.group_id,
            member_candidate_ids=g.member_candidate_ids,
            surviving_mask=g.surviving_mask,
            removed_region_ids=tuple(sorted(x for x in rem_set if x != g.group_id)),
            suppression_confidence=g.suppression_confidence,
            suppression_breakdown=g.suppression_breakdown,
            geometry_metadata=g.geometry_metadata,
        )
        for g in survivors_out
    )

    combined = np.zeros(regions[0].grouped_mask.shape, dtype=bool)
    for g in survivors:
        combined = np.logical_or(combined, g.surviving_mask)

    removed_t = tuple(sorted(rem_set))
    return survivors, removed_t, np.ascontiguousarray(combined)
