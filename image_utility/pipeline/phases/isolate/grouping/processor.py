"""Grouping-only orchestration: relationships → affinity → groups."""

from __future__ import annotations

import logging

from .affinity import pair_affinity_table
from .config import GroupingConfig, load_grouping_config
from .contracts import GroupingInput, GroupingMetadata, GroupingResult
from .debug.writers import write_grouping_debug
from .grouping import build_groups
from .relationships import build_pairwise_relationships

LOGGER = logging.getLogger(__name__)


class GroupingProcessor:
    """Semantic region association — not suppression or mask destruction."""

    def __init__(self, cfg: GroupingConfig | None = None) -> None:
        self._cfg = cfg or load_grouping_config()

    @property
    def config(self) -> GroupingConfig:
        return self._cfg

    def run(self, inp: GroupingInput, rgb, stem: str) -> GroupingResult:
        if not inp.proposals:
            raise ValueError("grouping stage requires at least one candidate")

        rels = build_pairwise_relationships(
            inp.proposals,
            image_hw=inp.image_hw,
            eps=self._cfg.math_epsilon,
        )
        table = pair_affinity_table(rels, self._cfg)
        LOGGER.info("[grouping] computed semantic affinities")

        groups = build_groups(inp.proposals, table, self._cfg)
        LOGGER.info("[grouping] generated grouped candidates=%d", len(groups))

        for g in groups:
            if len(g.member_candidate_ids) > 1:
                joined = ",".join(str(x) for x in g.member_candidate_ids)
                LOGGER.debug("[grouping] merged candidates=%s", joined)

        pair_flat = tuple(
            (a, b, float(table[(a, b)][0]))
            for a, b in sorted(table.keys())
        )

        multi = sum(1 for g in groups if len(g.member_candidate_ids) > 1)
        top_c = groups[0].group_confidence if groups else 0.0
        sec_c = groups[1].group_confidence if len(groups) > 1 else 0.0
        eps = self._cfg.math_epsilon
        amb = (
            len(groups) > 1
            and top_c > eps
            and (sec_c / top_c) >= self._cfg.grouping_ambiguity_ratio
        )

        meta = GroupingMetadata(
            group_count=len(groups),
            candidate_count=len(inp.proposals),
            multi_member_group_count=int(multi),
            top_group_confidence=top_c,
            second_group_confidence=sec_c,
            grouping_ambiguity=amb,
        )

        if groups:
            LOGGER.info("[grouping] top group confidence=%.2f", top_c)
        if amb:
            LOGGER.info("[grouping] ambiguity preserved among top groups")

        result = GroupingResult(
            groups=groups,
            relationships=rels,
            pair_affinities=pair_flat,
            metadata=meta,
        )
        write_grouping_debug(
            self._cfg,
            stem=stem,
            rgb=rgb,
            result=result,
            candidate_centroids={c.candidate_id: c.features.centroid_xy for c in inp.proposals},
        )
        return result
