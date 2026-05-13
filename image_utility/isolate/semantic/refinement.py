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


def _primary_component_feature(
    ranked: list[ComponentFeatures],
    cfg: IsolateConfig,
) -> ComponentFeatures | None:
    if not ranked:
        return None
    for f in ranked:
        if f.area >= cfg.min_component_area:
            return f
    return ranked[0]


def should_activate_semantic_refinement(
    stats: np.ndarray,
    alpha: NDArray[np.uint8],
    ranked: list[ComponentFeatures],
    cfg: IsolateConfig,
) -> tuple[bool, dict[str, Any]]:
    """
    Decide whether to run SAM semantic refinement; return activation metadata for debug/log.

    ``stats`` rows: label 0 = background, ≥1 foreground CCs. A **single** merged product capture is
    ``stats.shape[0] == 2`` and must not be rejected solely for component count.
    """
    n = stats.shape[0]
    n_fg = max(0, n - 1)
    base_meta: dict[str, Any] = {"reason": None, "n_foreground_cc": n_fg}

    if not cfg.semantic_refinement_enabled:
        return False, {**base_meta, "inactive": "semantic_disabled"}
    if not is_sam_available(cfg):
        return False, {**base_meta, "inactive": "sam_unavailable"}
    if not ranked:
        return False, {**base_meta, "inactive": "empty_ranked"}

    fg_area = int(np.count_nonzero(alpha > cfg.alpha_visibility_threshold))
    ih, iw = alpha.shape[:2]
    image_area = max(ih * iw, 1)
    fg_ratio = fg_area / float(image_area)

    total_cc = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)) if n > 1 else 0
    min_abs = max(
        cfg.min_component_area,
        int(total_cc * cfg.semantic_trigger_large_area_ratio),
    )
    large_ids = [i for i in range(1, n) if int(stats[i, cv2.CC_STAT_AREA]) >= min_abs]
    large_n = len(large_ids)

    viable = [f for f in ranked if f.area >= cfg.min_component_area and f.confidence > 0]
    best_conf = viable[0].confidence if viable else 0.0
    second_conf = viable[1].confidence if len(viable) > 1 else 0.0
    ambiguity = second_conf / max(best_conf, cfg.math_epsilon)

    top_borders = [f.border_contact_ratio for f in ranked[:2] if f.area >= cfg.min_component_area]

    base_signals: dict[str, Any] = {
        "n_foreground_cc": n_fg,
        "fg_area_ratio": round(fg_ratio, 4),
        "v2_best_confidence": round(best_conf, 4),
        "large_region_count": large_n,
    }

    def accept(reason: str, **signal_kw: float) -> tuple[bool, dict[str, Any]]:
        signals = {**base_signals, **{k: round(float(v), 4) for k, v in signal_kw.items()}}
        meta: dict[str, Any] = {**base_meta, "reason": reason, "signals": signals}
        LOGGER.info("[isolate] semantic trigger: %s", reason)
        shown = 0
        for k, v in signal_kw.items():
            if shown >= 3:
                break
            LOGGER.info("[isolate] semantic trigger %s=%.2f", k, float(v))
            shown += 1
        return True, meta

    if best_conf < cfg.semantic_trigger_v2_conf_below:
        return accept("ambiguous_confidence", v2_best_confidence=best_conf)

    if len(viable) >= 2 and ambiguity >= cfg.semantic_trigger_second_ratio_min:
        return accept("ambiguous_confidence", second_over_best=ambiguity)

    if large_n >= cfg.semantic_trigger_min_large_regions:
        return accept("multi_large_regions", large_n=float(large_n))

    if (
        len(top_borders) >= 2
        and top_borders[0] >= cfg.semantic_trigger_border_min
        and top_borders[1] >= cfg.semantic_trigger_border_min
    ):
        return accept("border_conflict", border_top0=top_borders[0], border_top1=top_borders[1])

    if large_n >= 2 and fg_ratio >= cfg.semantic_trigger_fg_area_ratio:
        return accept("multi_large_regions", bulky_fg_ratio=fg_ratio)

    if n_fg == 1 and fg_ratio >= cfg.semantic_trigger_single_fg_ratio_min:
        feat = _primary_component_feature(ranked, cfg)
        if feat is not None:
            _bx, _by, bw, bh = feat.bbox
            fill_ratio = float(feat.area) / max(float(bw * bh), 1.0)
            geom = {
                "border_contact": float(feat.border_contact_ratio),
                "solidity": float(feat.solidity),
                "elongation": float(feat.elongation),
                "fill_ratio": float(fill_ratio),
                "complexity": float(feat.complexity),
            }
            triggered: list[str] = []
            if feat.border_contact_ratio >= cfg.semantic_trigger_single_border_contact:
                triggered.append("border_contact")
            if feat.solidity <= cfg.semantic_trigger_single_solidity_max:
                triggered.append("solidity")
            if feat.elongation >= cfg.semantic_trigger_single_elongation_min:
                triggered.append("elongation")
            if fill_ratio < cfg.semantic_trigger_single_fill_ratio_min:
                triggered.append("fill_ratio")
            if feat.complexity >= cfg.semantic_trigger_single_complexity_min:
                triggered.append("complexity")

            if triggered:
                spread_only = set(triggered) <= {"fill_ratio", "complexity"}
                reason = "fg_spread_conflict" if spread_only else "suspicious_single_region"
                signals = {**base_signals, **{k: round(v, 4) for k, v in geom.items()}}
                meta = {
                    **base_meta,
                    "reason": reason,
                    "triggered_rules": triggered,
                    "signals": signals,
                }
                LOGGER.info("[isolate] semantic trigger: %s", reason)
                LOGGER.info("[isolate] semantic trigger rules=%s", ",".join(triggered))
                LOGGER.info(
                    "[isolate] semantic trigger border_contact=%.2f solidity=%.2f elongation=%.2f",
                    geom["border_contact"],
                    geom["solidity"],
                    geom["elongation"],
                )
                return True, meta

    return False, {**base_meta, "signals": base_signals}


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
