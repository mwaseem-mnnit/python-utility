"""Decomposition-only orchestration (rembg → topology → SAM → normalize)."""

from __future__ import annotations

import logging

import numpy as np

from .config import DecompositionConfig, load_decomposition_config
from .connected_regions import (
    analyze_components,
    binary_from_alpha,
    extract_connected_regions,
    morph_pre_close,
)
from .contracts import DecompositionMetadata, DecompositionResult
from .debug.writers import write_decomposition_debug
from .models import SamDecomposer
from .normalize import (
    connected_regions_to_candidates,
    deduplicate_conservative,
    morph_post_open_binary,
    raw_dicts_to_candidates,
    renumber_candidates,
)
from .rembg_extract import extract_foreground

LOGGER = logging.getLogger(__name__)


class DecompositionProcessor:
    """
    Runs the decomposition stage in isolation—proposals only, no ranking or selection.

    Implements the flow from ``PHASE_1_ISOLATE_ARCHITECTURE.md`` / ``ISOLATE_DECOMPOSITION_AUTHORITY.md``.
    """

    def __init__(self, cfg: DecompositionConfig | None = None) -> None:
        self._cfg = cfg or load_decomposition_config()
        self._sam = SamDecomposer(self._cfg)

    @property
    def config(self) -> DecompositionConfig:
        return self._cfg

    def run(self, rgb: np.ndarray, stem: str) -> DecompositionResult:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("decomposition expects RGB uint8 H×W×3")

        boot = extract_foreground(rgb, self._cfg)
        if not np.any(boot.alpha > self._cfg.alpha_visibility_threshold):
            raise OSError("decomposition: empty foreground from rembg")

        LOGGER.info("[decomposition] rembg foreground extracted")

        bin_m = binary_from_alpha(boot.alpha, self._cfg.alpha_visibility_threshold)
        bin_m = morph_pre_close(bin_m, self._cfg.morph_pre_close_size)
        bin_m = morph_post_open_binary(bin_m, self._cfg.morph_post_open_size)
        labels, stats, centroids = analyze_components(bin_m)
        if stats.shape[0] <= 1:
            raise OSError("decomposition: no connected foreground components")

        connected = extract_connected_regions(labels, stats, centroids, self._cfg)
        LOGGER.info("[decomposition] connected regions=%d", len(connected))

        sam_raw: list[dict] = []
        notes: list[str] = []
        if self._sam.is_available():
            sam_raw = self._sam.generate_raw(rgb)
            LOGGER.info("[decomposition] semantic masks=%d", len(sam_raw))
            if not sam_raw:
                notes.append("sam_returned_empty")
        else:
            notes.append("sam_unavailable_or_disabled")

        ih, iw = rgb.shape[:2]
        sam_candidates = raw_dicts_to_candidates(sam_raw, (ih, iw), self._cfg)
        cc_candidates = connected_regions_to_candidates(labels, self._cfg)

        merged = renumber_candidates(sam_candidates + cc_candidates)
        normalized = deduplicate_conservative(merged, self._cfg)
        LOGGER.info("[decomposition] candidate normalization complete (%d candidates)", len(normalized))

        alpha_cands = tuple(
            np.where(c.mask, boot.alpha, 0).astype(np.uint8) for c in normalized
        )

        meta = DecompositionMetadata(
            rembg_model=self._cfg.rembg_model_name,
            morph_pre_close=self._cfg.morph_pre_close_size,
            sam_enabled=self._sam.is_available(),
            sam_raw_mask_count=len(sam_raw),
            normalized_candidate_count=len(normalized),
            connected_region_count=len(connected),
            alpha_candidate_count=len(alpha_cands),
            notes=tuple(notes),
        )

        result = DecompositionResult(
            base_rgba=boot.rgba,
            base_alpha=boot.alpha,
            cc_labels=labels,
            cc_stats=stats,
            cc_centroids=centroids,
            connected_regions=connected,
            semantic_candidates=tuple(normalized),
            alpha_candidates=alpha_cands,
            metadata=meta,
        )

        write_decomposition_debug(self._cfg, stem=stem, rgb=rgb, result=result)
        return result
