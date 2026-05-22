"""Ranking-only orchestration: features → scores → ordered evidence."""

from __future__ import annotations

import logging

import numpy as np

from .config import RankingConfig, load_ranking_config
from .contracts import (
    RankingInput,
    RankingMetadata,
    RankingResult,
)
from .debug.writers import write_ranking_debug
from .features import extract_all_features
from .scoring import score_candidate

LOGGER = logging.getLogger(__name__)


class RankingProcessor:
    """
    Produces soft semantic confidence for every proposal — no suppression or grouping.
    """

    def __init__(self, cfg: RankingConfig | None = None) -> None:
        self._cfg = cfg or load_ranking_config()

    @property
    def config(self) -> RankingConfig:
        return self._cfg

    def run(self, inp: RankingInput, rgb: np.ndarray, stem: str) -> RankingResult:
        if not inp.proposals:
            raise ValueError("ranking stage requires at least one proposal")

        feats_tuple = extract_all_features(inp.proposals, inp.base_alpha, self._cfg)
        LOGGER.info("[ranking] extracted features for candidates=%d", len(inp.proposals))

        scored = [
            score_candidate(p, f, self._cfg)
            for p, f in zip(inp.proposals, feats_tuple)
        ]
        ranked_list = sorted(scored, key=lambda r: r.confidence, reverse=True)
        ranked = tuple(ranked_list)

        top_c = ranked[0].confidence
        second_c = ranked[1].confidence if len(ranked) > 1 else 0.0
        sep = (top_c - second_c) if len(ranked) > 1 else top_c
        amb = (
            len(ranked) > 1
            and top_c > self._cfg.math_epsilon
            and (second_c / top_c) >= self._cfg.ambiguity_ratio_threshold
        )

        meta = RankingMetadata(
            candidate_count=len(ranked),
            ambiguity_detected=amb,
            top_confidence=top_c,
            second_confidence=second_c,
            confidence_separation=sep,
        )

        result = RankingResult(ranked=ranked, metadata=meta)

        LOGGER.info("[ranking] generated semantic confidence")
        if amb:
            LOGGER.info("[ranking] ambiguity detected (top=%.3f second=%.3f)", top_c, second_c)
        LOGGER.info("[ranking] top candidate confidence=%.2f", top_c)

        write_ranking_debug(self._cfg, stem=stem, rgb=rgb, base_alpha=inp.base_alpha, result=result)
        return result
