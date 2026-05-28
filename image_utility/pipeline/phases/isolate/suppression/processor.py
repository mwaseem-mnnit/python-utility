"""Suppression orchestration."""

from __future__ import annotations

import logging

import numpy as np

from .config import SuppressionConfig, load_suppression_config
from .contracts import (
    SuppressionInput,
    SuppressionMetadata,
    SuppressionResult,
)
from .debug.writers import write_suppression_debug
from .scoring import score_all_regions
from .suppression import suppress_groups

LOGGER = logging.getLogger(__name__)


class SuppressionProcessor:
    """Semantic cleanup — no feathering or inpainting."""

    def __init__(self, cfg: SuppressionConfig | None = None) -> None:
        self._cfg = cfg or load_suppression_config()

    @property
    def config(self) -> SuppressionConfig:
        return self._cfg

    def run(self, inp: SuppressionInput, rgb: np.ndarray, stem: str) -> SuppressionResult:
        if not inp.regions:
            raise ValueError("suppression requires at least one grouped region")

        LOGGER.info("[suppression] analyzed grouped regions=%d", len(inp.regions))
        scores = score_all_regions(inp.regions, self._cfg)

        survivors, removed_ids, combined = suppress_groups(inp.regions, scores, self._cfg)

        if removed_ids:
            LOGGER.info("[suppression] removed artifact regions=%d", len(removed_ids))
        LOGGER.info("[suppression] preserved semantic groups=%d", len(survivors))

        # Area-weighted global confidence across survivors (robust aggregation)
        if survivors:
            wsum = sum(float(np.count_nonzero(g.surviving_mask)) for g in survivors)
            if wsum <= self._cfg.math_epsilon:
                g_conf = float(self._cfg.confidence_floor)
            else:
                g_conf = sum(
                    float(np.count_nonzero(g.surviving_mask)) * g.suppression_confidence
                    for g in survivors
                ) / wsum
            g_conf = float(max(self._cfg.confidence_floor, min(1.0, g_conf)))
        else:
            g_conf = float(self._cfg.confidence_floor)

        LOGGER.info("[suppression] suppression confidence=%.2f", g_conf)

        meta = SuppressionMetadata(
            analyzed_group_count=len(inp.regions),
            removed_group_count=len(removed_ids),
            surviving_group_count=len(survivors),
            removed_group_ids=removed_ids,
            global_suppression_confidence=g_conf,
        )

        result = SuppressionResult(
            surviving_groups=survivors,
            scores=scores,
            combined_survivor_mask=combined,
            metadata=meta,
        )
        write_suppression_debug(self._cfg, stem=stem, rgb=rgb, inp=inp, result=result)
        return result
