"""Filtering stage orchestration."""

from __future__ import annotations

import logging

import numpy as np

from .config import FilteringConfig, load_filtering_config
from .contracts import (
    FilteringInput,
    FilteringMetadata,
    FilteringProposal,
    FilteringResult,
    FilteringScore,
    ScoredFilteringProposal,
)
from .debug import write_filtering_debug
from .dedup import deduplicate_proposals
from .scoring import score_proposal

LOGGER = logging.getLogger(__name__)


def _accept_and_reject(
    scored: tuple[ScoredFilteringProposal, ...],
    cfg: FilteringConfig,
) -> tuple[
    tuple[FilteringProposal, ...],
    tuple[FilteringProposal, ...],
    bool,
]:
    rejected_list: list = []
    accepted_list: list = []
    for sp in scored:
        if sp.filtering_score.rejection_likelihood >= cfg.reject_threshold:
            rejected_list.append(sp.proposal)
        else:
            accepted_list.append(sp.proposal)

    fallback = False
    if scored and not accepted_list:
        best = max(scored, key=lambda s: s.filtering_score.validity_score)
        fallback = True
        accepted_list = [best.proposal]
        rejected_list = [
            sp.proposal for sp in scored if sp.proposal.candidate_id != best.proposal.candidate_id
        ]

    return tuple(accepted_list), tuple(rejected_list), fallback


class FilteringProcessor:
    """Proposal gating — no ranking, merging, or alpha feathering."""

    def __init__(self, cfg: FilteringConfig | None = None) -> None:
        self._cfg = cfg or load_filtering_config()

    def run(self, inp: FilteringInput, rgb: np.ndarray, stem: str) -> FilteringResult:
        if inp.proposals and rgb.shape[:2] != tuple(inp.image_hw):
            raise ValueError("filtering RGB dimensions must match FilteringInput.image_hw")

        if not inp.proposals:
            meta = FilteringMetadata(
                input_count=0,
                accepted_count=0,
                rejected_count=0,
                all_rejected_fallback=False,
                dedup_removed_count=0,
                post_dedup_count=0,
            )
            result = FilteringResult(
                accepted=(),
                rejected=(),
                scored=(),
                metadata=meta,
            )
            write_filtering_debug(self._cfg, stem=stem, rgb=rgb, inp=inp, result=result)
            return result

        original_count = len(inp.proposals)
        LOGGER.info("[filtering] input proposals=%d", original_count)

        # Step 1: deduplicate near-identical masks before scoring
        proposals, dedup_removed = deduplicate_proposals(inp.proposals, self._cfg)
        post_dedup_count = len(proposals)

        LOGGER.info("[filtering] post-dedup proposals=%d (removed %d duplicates)", post_dedup_count, dedup_removed)

        # Step 2: score each surviving proposal
        scores_list: list[FilteringScore] = []
        for p in proposals:
            scores_list.append(score_proposal(p, rgb, cfg=self._cfg))

        scores_t = tuple(scores_list)
        scored = tuple(
            ScoredFilteringProposal(proposal=p, filtering_score=s)
            for p, s in zip(proposals, scores_t)
        )

        # Step 3: accept/reject gate
        acc, rej, fallback = _accept_and_reject(scored, self._cfg)

        if fallback:
            LOGGER.info("[filtering] conservative salvage — retaining best-validity proposal only")

        LOGGER.info("[filtering] removed artifact regions=%d", len(rej))
        LOGGER.info("[filtering] accepted proposals=%d", len(acc))
        if scored:
            topv = max(scored, key=lambda z: z.filtering_score.validity_score)
            LOGGER.info(
                "[filtering] filtering confidence=%.2f (top validity id=%d)",
                topv.filtering_score.validity_score,
                topv.proposal.candidate_id,
            )

        meta = FilteringMetadata(
            input_count=original_count,
            accepted_count=len(acc),
            rejected_count=len(rej),
            all_rejected_fallback=fallback,
            dedup_removed_count=dedup_removed,
            post_dedup_count=post_dedup_count,
        )
        result = FilteringResult(accepted=acc, rejected=rej, scored=scored, metadata=meta)
        write_filtering_debug(self._cfg, stem=stem, rgb=rgb, inp=inp, result=result)
        return result
