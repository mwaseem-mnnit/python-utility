"""Construct merged groups from affinity (union-find; non-destructive masks)."""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from .config import GroupingConfig
from .contracts import GroupedCandidate, GroupingCandidateInput

LOGGER = logging.getLogger(__name__)
BoolMask = NDArray[np.bool_]


class _UnionFind:
    def __init__(self, keys: list[int]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, x: int) -> int:
        p = self._parent[x]
        if p != x:
            self._parent[x] = self.find(p)
        return self._parent[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _geometric_mean(values: list[float], eps: float) -> float:
    if not values:
        return 0.0
    logv = [math.log(max(v, eps)) for v in values]
    return float(math.exp(sum(logv) / len(logv)))


def build_groups(
    candidates: tuple[GroupingCandidateInput, ...],
    pair_affinities: dict[tuple[int, int], tuple[float, dict[str, float]]],
    cfg: GroupingConfig,
) -> tuple[GroupedCandidate, ...]:
    ids = [c.candidate_id for c in candidates]
    by_id = {c.candidate_id: c for c in candidates}
    uf = _UnionFind(ids)

    for (i, j), (aff, _) in pair_affinities.items():
        if aff >= cfg.merge_affinity_threshold:
            uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for cid in ids:
        r = uf.find(cid)
        clusters.setdefault(r, []).append(cid)

    groups: list[GroupedCandidate] = []
    shape = next(iter(candidates)).mask.shape
    for members in sorted(clusters.values(), key=lambda m: min(m)):
        members_t = tuple(sorted(members))
        merged: BoolMask = np.zeros(shape, dtype=bool)
        confidences: list[float] = []
        pair_keys: list[tuple[int, int, float]] = []
        aff_parts: dict[str, list[float]] = {}

        for mid in members:
            mask = by_id[mid].mask
            LOGGER.debug(
                "[group-debug] candidate=%s nonzero=%d dtype=%s shape=%s",
                mid,
                int(np.count_nonzero(mask)),
                mask.dtype,
                mask.shape,
            )
            merged = np.logical_or(merged, by_id[mid].mask)
            confidences.append(max(cfg.confidence_floor, float(by_id[mid].confidence)))

        if len(members_t) < 2:
            breakdown = {"internal_pair_affinity_mean": 1.0}
            pair_keys = []
        else:
            for a in range(len(members_t)):
                for b in range(a + 1, len(members_t)):
                    key = (min(members_t[a], members_t[b]), max(members_t[a], members_t[b]))
                    aff, bd = pair_affinities.get(key, (0.0, {}))
                    pair_keys.append((members_t[a], members_t[b], float(aff)))
                    for k, v in bd.items():
                        if k == "affinity":
                            continue
                        aff_parts.setdefault(k, []).append(float(v))

            breakdown = {k: float(sum(v) / len(v)) for k, v in aff_parts.items()}
            breakdown["internal_pair_affinity_mean"] = float(
                sum(p[2] for p in pair_keys) / max(len(pair_keys), 1)
            )

        geom_area = int(np.count_nonzero(merged))
        ys, xs = np.where(merged)
        if geom_area == 0:
            cx, cy = 0.0, 0.0
            bbox = (0, 0, 0, 0)
        else:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())
            bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

        gmean_conf = _geometric_mean(confidences, cfg.math_epsilon)
        cohesion = float(
            breakdown.get("internal_pair_affinity_mean", cfg.confidence_floor)
        )
        group_confidence = float(
            max(
                cfg.confidence_floor,
                min(1.0, gmean_conf * (0.65 + 0.35 * cohesion)),
            )
        )

        geometry_metadata: dict[str, float | int] = {
            "pixel_area": geom_area,
            "member_count": len(members_t),
            "centroid_x": cx,
            "centroid_y": cy,
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
        }

        groups.append(
            GroupedCandidate(
                group_id=0,
                member_candidate_ids=members_t,
                grouped_mask=merged,
                group_confidence=group_confidence,
                affinity_breakdown=breakdown,
                relationship_pairs=tuple(pair_keys),
                geometry_metadata=geometry_metadata,
            )
        )

    groups.sort(key=lambda g: g.group_confidence, reverse=True)
    for idx, g in enumerate(groups):
        # re-assign stable ids by confidence rank
        groups[idx] = GroupedCandidate(
            group_id=idx,
            member_candidate_ids=g.member_candidate_ids,
            grouped_mask=g.grouped_mask,
            group_confidence=g.group_confidence,
            affinity_breakdown=g.affinity_breakdown,
            relationship_pairs=g.relationship_pairs,
            geometry_metadata=g.geometry_metadata,
        )

    return tuple(groups)
