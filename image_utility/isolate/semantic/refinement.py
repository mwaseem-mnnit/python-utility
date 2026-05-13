"""Conditional SAM refinement: triggers, orchestration, conservative fallback."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ..components import ComponentFeatures
from ..config import IsolateConfig
from . import masks as mask_ops
from . import ranking as rank_ops
from .debug import write_semantic_debug
from .masks import PreparedMask
from .sam_runner import generate_raw_masks, is_sam_available

LOGGER = logging.getLogger(__name__)


def should_activate_semantic_refinement(
    stats: np.ndarray,
    alpha: NDArray[np.uint8],
    ranked: list[ComponentFeatures],
    cfg: IsolateConfig,
) -> bool:
    if not cfg.semantic_refinement_enabled:
        return False
    if not is_sam_available(cfg):
        return False

    n = stats.shape[0]
    if n <= 2 or not ranked:
        return False

    fg_area = int(np.count_nonzero(alpha > cfg.alpha_visibility_threshold))
    ih, iw = alpha.shape[:2]
    image_area = max(ih * iw, 1)
    fg_ratio = fg_area / float(image_area)

    total_cc = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n))
    min_abs = max(
        cfg.min_component_area,
        int(total_cc * cfg.semantic_trigger_large_area_ratio),
    )

    large_ids = [
        i
        for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_abs
    ]
    large_n = len(large_ids)

    viable = [f for f in ranked if f.area >= cfg.min_component_area and f.confidence > 0]
    best_conf = viable[0].confidence if viable else 0.0
    second_conf = viable[1].confidence if len(viable) > 1 else 0.0
    ambiguity = second_conf / max(best_conf, cfg.math_epsilon)

    top_borders = [f.border_contact_ratio for f in ranked[:2] if f.area >= cfg.min_component_area]

    # Multiple large connected regions
    if large_n >= cfg.semantic_trigger_min_large_regions:
        return True

    # Low confidence separation (ambiguous winner)
    if best_conf < cfg.semantic_trigger_v2_conf_below:
        return True

    if len(viable) >= 2 and ambiguity >= cfg.semantic_trigger_second_ratio_min:
        return True

    # Border-contact conflict between top two plausible regions
    if (
        len(top_borders) >= 2
        and top_borders[0] >= cfg.semantic_trigger_border_min
        and top_borders[1] >= cfg.semantic_trigger_border_min
    ):
        return True

    # Bulky multi-part foreground
    if large_n >= 2 and fg_ratio >= cfg.semantic_trigger_fg_area_ratio:
        return True

    return False


def apply_semantic_refinement(
    rgb: NDArray[np.uint8],
    alpha: NDArray[np.uint8],
    labels: NDArray[np.int32],
    keep_heuristic: int,
    cfg: IsolateConfig,
    *,
    stem: str,
) -> tuple[NDArray[np.uint8] | None, dict[str, Any]]:
    """
    Return refined alpha (uint8) or ``(None, meta)`` to fall back to CC/heuristic path.

    Refined alpha preserves original alpha **values** inside the chosen SAM mask.
    """
    meta: dict[str, Any] = {"activated": True}
    try:
        fg = mask_ops.foreground_bool(alpha, cfg.alpha_visibility_threshold)
        heur = labels == keep_heuristic

        raw = generate_raw_masks(rgb, cfg)
        if raw is None:
            meta["reason"] = "sam_unavailable_or_failed"
            return None, meta

        meta["raw_mask_count"] = len(raw)
        LOGGER.info("[isolate] generated %d semantic masks", len(raw))

        candidates = mask_ops.prepare_sam_candidates(raw, fg, cfg)
        if not candidates:
            meta["reason"] = "no_sam_candidates"
            return None, meta

        ih, iw = alpha.shape[:2]
        prepared_list: list[PreparedMask] = candidates
        scored = rank_ops.rank_semantic_regions(prepared_list, fg, (ih, iw), cfg, heuristic_mask=heur)
        meta["candidate_count"] = len(scored)

        if not scored:
            meta["reason"] = "rank_empty"
            meta["rejection_detail"] = "no_ranked_regions"
            return None, meta

        best = scored[0]
        if best.confidence < cfg.semantic_catastrophic_confidence:
            meta["reason"] = "semantic_catastrophic_low_confidence"
            meta["rejection_detail"] = "best_below_catastrophic_floor"
            meta["best_confidence"] = float(best.confidence)
            meta["catastrophic_floor"] = float(cfg.semantic_catastrophic_confidence)
            return None, meta

        agree = float(best.breakdown.get("heuristic_agreement", 0.0))
        meta["heuristic_agreement"] = agree
        meta["selected_heuristic_agreement"] = agree
        meta["selected_semantic_dominance"] = float(best.breakdown.get("semantic_dominance", 0.0))
        meta["semantic_ranking"] = [
            {
                "region_id": int(s.region_id),
                "confidence": round(float(s.confidence), 4),
                "semantic_dominance": round(float(s.breakdown.get("semantic_dominance", 0.0)), 4),
                "breakdown": {k: float(v) for k, v in s.breakdown.items()},
            }
            for s in scored
        ]
        meta["fallback_reason"] = None
        meta["authoritative"] = True

        out_alpha = np.where(best.mask, alpha, 0).astype(np.uint8)
        meta["selected_semantic_confidence"] = float(best.confidence)
        meta["selected_region_id"] = int(best.region_id)

        write_semantic_debug(cfg, stem=stem, rgb=rgb, scored=scored, selected=best)

        LOGGER.info("[isolate] heuristic agreement=%.2f (soft signal)", agree)
        LOGGER.info("[isolate] semantic dominance selected region=%d", best.region_id)
        LOGGER.info("[isolate] selected semantic region confidence=%.2f", best.confidence)
        LOGGER.info("[isolate] semantic refinement authoritative")

        return out_alpha, meta
    except Exception as e:
        LOGGER.warning("[isolate] semantic refinement internal error: %s", e)
        return None, {**meta, "reason": "exception", "error": str(e)}
