"""Isolate pipeline orchestration (high-level steps only)."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext
from image_utility.pipeline.phases.isolate.decomposition import DecompositionProcessor
from image_utility.pipeline.phases.isolate.grouping import (
    GroupingCandidateInput,
    GroupingFeatureVector,
    GroupingInput,
    GroupingProcessor,
    GroupingRankingSnapshot,
)
from image_utility.pipeline.phases.isolate.ranking import (
    RankingInput,
    RankingMaskInput,
    RankingProcessor,
    RankingResult,
)

from .cleanup import (
    morphological_post_open,
    strip_small_fragments,
)
from .components import (
    apply_kept_label_to_alpha,
    select_best_component,
)
from .config import IsolateConfig, load_isolate_config
from .debug import write_isolate_debug
from .refinement import compose_isolated_rgba, refine_alpha_soft
from .semantic.refinement import apply_semantic_refinement, should_activate_semantic_refinement

LOGGER = logging.getLogger(__name__)


def _build_grouping_input(rank_res: RankingResult, base_alpha: np.ndarray) -> GroupingInput:
    """Copy ranking evidence into grouping contracts (no grouping→ranking imports)."""

    props: list[GroupingCandidateInput] = []
    for r in rank_res.ranked:
        f = r.features
        props.append(
            GroupingCandidateInput(
                candidate_id=r.candidate_id,
                source=r.source,
                mask=r.mask,
                confidence=float(r.confidence),
                features=GroupingFeatureVector(
                    area=f.area,
                    relative_area=float(f.relative_area),
                    centroid_xy=tuple(f.centroid_xy),
                    bbox_xywh=tuple(f.bbox_xywh),
                    bbox_fill_ratio=float(f.bbox_fill_ratio),
                    solidity=float(f.solidity),
                    elongation=float(f.elongation),
                    border_contact_ratio=float(f.border_contact_ratio),
                    contour_complexity=float(f.contour_complexity),
                    occupancy_ratio=float(f.occupancy_ratio),
                    center_distance_norm=float(f.center_distance_norm),
                    sam_predicted_iou=float(f.sam_predicted_iou),
                    sam_stability=float(f.sam_stability),
                    overlap_rembg_fg=float(f.overlap_rembg_fg),
                ),
                score_breakdown=dict(r.score_breakdown),
            )
        )
    h, w = base_alpha.shape[:2]
    snap = GroupingRankingSnapshot(
        candidate_count=rank_res.metadata.candidate_count,
        ambiguity_detected=rank_res.metadata.ambiguity_detected,
        top_confidence=float(rank_res.metadata.top_confidence),
        second_confidence=float(rank_res.metadata.second_confidence),
        confidence_separation=float(rank_res.metadata.confidence_separation),
    )
    return GroupingInput(
        proposals=tuple(props),
        base_alpha=base_alpha,
        image_hw=(h, w),
        ranking_snapshot=snap,
    )


def _cc_label_best_overlap(labels: np.ndarray, stats: np.ndarray, mask: np.ndarray) -> int:
    best_lab, best_o = 1, -1
    for li in range(1, stats.shape[0]):
        m = labels == li
        o = int(np.count_nonzero(np.logical_and(m, mask)))
        if o > best_o:
            best_o = o
            best_lab = li
    return best_lab


def process_isolate(
    context: PipelineContext,
    *,
    cfg: IsolateConfig | None = None,
) -> PipelineContext:
    """
    Populate ``context.current_rgba`` and ``context.alpha_mask``.

    Raises ``OSError`` for recoverable failures so the runner can skip the file.
    """
    cfg = cfg or load_isolate_config()
    stem = context.input_path.stem
    name = context.input_path.name

    rgb = context.current_image
    if rgb is None:
        raise OSError("isolate requires current_image (RGB)")

    decomp_proc = DecompositionProcessor()
    try:
        decomp = decomp_proc.run(rgb, stem=stem)
    except OSError:
        raise
    except Exception as exc:
        LOGGER.warning("[isolate] decomposition failed: %s", exc)
        raise OSError(f"decomposition failed: {exc}") from exc

    context.metadata["decomposition_result"] = decomp
    context.debug["decomposition"] = {
        "connected_region_count": len(decomp.connected_regions),
        "semantic_candidate_count": len(decomp.semantic_candidates),
        "sam_raw_mask_count": decomp.metadata.sam_raw_mask_count,
        "alpha_candidate_count": decomp.metadata.alpha_candidate_count,
        "notes": list(decomp.metadata.notes),
    }

    stop_after = os.getenv("ISOLATE_STOP_AFTER_STAGE", "").strip().lower()
    if stop_after == "decomposition":
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(decomp.base_alpha)
        context.metadata["isolate_stopped_after"] = "decomposition"
        context.debug["isolate_stopped_after"] = "decomposition"
        LOGGER.info("[isolate] stop after decomposition — downstream isolate substages skipped")
        LOGGER.info("[isolate] complete %s (decomposition-only)", name)
        return context

    rgba = decomp.base_rgba
    alpha = decomp.base_alpha
    labels = decomp.cc_labels
    stats = decomp.cc_stats
    centroids = decomp.cc_centroids

    LOGGER.info("[isolate] segmented %s", name)

    if not np.any(alpha > cfg.alpha_visibility_threshold):
        raise OSError("segmentation collapse: empty alpha")

    if stats.shape[0] <= 1:
        raise OSError("no foreground components detected")

    proposals = tuple(
        RankingMaskInput(
            candidate_id=c.candidate_id,
            mask=c.mask,
            source=c.source,
            predicted_iou=c.predicted_iou,
            stability_score=c.stability_score,
            area=c.area,
        )
        for c in decomp.semantic_candidates
    )

    ranking_fallback = False
    rank_res = None
    if proposals:
        h, w = alpha.shape[:2]
        rank_in = RankingInput(proposals=proposals, base_alpha=alpha, image_hw=(h, w))
        try:
            rank_res = RankingProcessor().run(rank_in, rgb, stem)
        except Exception as exc:
            LOGGER.warning("[isolate] ranking stage failed (%s); falling back to legacy CC selection", exc)
            ranking_fallback = True
        else:
            context.metadata["ranking_result"] = rank_res
            context.debug["ranking"] = {
                "candidate_count": rank_res.metadata.candidate_count,
                "ambiguity_detected": rank_res.metadata.ambiguity_detected,
                "top_confidence": round(rank_res.metadata.top_confidence, 4),
                "second_confidence": round(rank_res.metadata.second_confidence, 4),
                "confidence_separation": round(rank_res.metadata.confidence_separation, 4),
            }
            context.debug["isolate_ranking_top"] = [
                {
                    "candidate_id": r.candidate_id,
                    "source": r.source,
                    "confidence": round(r.confidence, 4),
                    "score_breakdown": {k: round(float(v), 4) for k, v in r.score_breakdown.items()},
                }
                for r in rank_res.ranked[:12]
            ]
    else:
        ranking_fallback = True
        LOGGER.warning("[isolate] no decomposition candidates; legacy CC selection")

    if stop_after == "ranking":
        if ranking_fallback or rank_res is None:
            raise OSError("ISOLATE_STOP_AFTER_STAGE=ranking requires a successful ranking run")
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(decomp.base_alpha)
        context.metadata["isolate_stopped_after"] = "ranking"
        context.debug["isolate_stopped_after"] = "ranking"
        LOGGER.info("[isolate] stop after ranking — grouping/suppression/refinement skipped")
        LOGGER.info("[isolate] complete %s (ranking-only)", name)
        return context

    group_res = None
    grouping_fallback = False
    if rank_res is not None:
        try:
            group_in = _build_grouping_input(rank_res, alpha)
            group_res = GroupingProcessor().run(group_in, rgb, stem)
        except Exception as exc:
            LOGGER.warning("[isolate] grouping stage failed (%s); using ranking-only selection", exc)
            grouping_fallback = True
    else:
        grouping_fallback = True

    if group_res is not None:
        context.metadata["grouping_result"] = group_res
        context.debug["grouping"] = {
            "group_count": group_res.metadata.group_count,
            "candidate_count": group_res.metadata.candidate_count,
            "multi_member_group_count": group_res.metadata.multi_member_group_count,
            "top_group_confidence": round(group_res.metadata.top_group_confidence, 4),
            "second_group_confidence": round(group_res.metadata.second_group_confidence, 4),
            "grouping_ambiguity": group_res.metadata.grouping_ambiguity,
        }
        context.debug["isolate_grouping_top"] = [
            {
                "group_id": g.group_id,
                "member_candidate_ids": list(g.member_candidate_ids),
                "group_confidence": round(g.group_confidence, 4),
                "affinity_breakdown": {k: round(float(v), 4) for k, v in g.affinity_breakdown.items()},
            }
            for g in group_res.groups[:12]
        ]

    if stop_after == "grouping":
        if ranking_fallback or rank_res is None or group_res is None or grouping_fallback:
            raise OSError(
                "ISOLATE_STOP_AFTER_STAGE=grouping requires successful ranking and grouping runs"
            )
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(decomp.base_alpha)
        context.metadata["isolate_stopped_after"] = "grouping"
        context.debug["isolate_stopped_after"] = "grouping"
        LOGGER.info("[isolate] stop after grouping — suppression/refinement skipped")
        LOGGER.info("[isolate] complete %s (grouping-only)", name)
        return context

    legacy_ranked = None
    keep = 1
    if group_res is not None and group_res.groups:
        top_g = group_res.groups[0]
        LOGGER.debug(
            "[debug-group] group_mask dtype=%s shape=%s nonzero=%d unique=%s",
            top_g.grouped_mask.dtype,
            top_g.grouped_mask.shape,
            int(np.count_nonzero(top_g.grouped_mask)),
            np.unique(top_g.grouped_mask)[:10],
        )
        masked_alpha = np.where(top_g.grouped_mask, 255, 0).astype(np.uint8)
        keep = _cc_label_best_overlap(labels, stats, top_g.grouped_mask)
    elif not ranking_fallback and rank_res is not None:
        top = rank_res.ranked[0]
        masked_alpha = np.where(top.mask, alpha, 0).astype(np.uint8)
        keep = _cc_label_best_overlap(labels, stats, top.mask)
    else:
        keep, legacy_ranked = select_best_component(labels, stats, centroids, cfg)
        if keep is None:
            raise OSError("could not select a product component")
        masked_alpha = apply_kept_label_to_alpha(alpha, labels, keep)

    n_fg = int(stats.shape[0] - 1)
    LOGGER.info("[isolate] ranked %d foreground components", n_fg)

    sel_area = int(np.count_nonzero(masked_alpha > cfg.alpha_visibility_threshold))
    best_feats = None
    if ranking_fallback and legacy_ranked:
        sel_area = int(stats[keep, cv2.CC_STAT_AREA])
        best_feats = next(f for f in legacy_ranked if f.label == keep)
        LOGGER.info("[isolate] selected component confidence=%.2f", best_feats.confidence)
        LOGGER.info("[isolate] selected label=%d area=%d", keep, sel_area)
        if cfg.v2_weight_border_contact > 0 and best_feats.border_contact_ratio > 0.02:
            LOGGER.info("[isolate] applied border contact penalty")
    elif group_res is not None and group_res.groups:
        top_g = group_res.groups[0]
        LOGGER.info(
            "[isolate] grouping-selected group_id=%d members=%s confidence=%.2f",
            top_g.group_id,
            ",".join(str(x) for x in top_g.member_candidate_ids),
            top_g.group_confidence,
        )
    elif rank_res is not None:
        LOGGER.info(
            "[isolate] ranking-selected candidate id=%d confidence=%.2f",
            rank_res.ranked[0].candidate_id,
            rank_res.ranked[0].confidence,
        )

    v3: dict[str, object] = {"used": False}
    activate_semantic = False
    activation_meta: dict = {}

    if ranking_fallback:
        activate_semantic, activation_meta = should_activate_semantic_refinement(
            stats, alpha, legacy_ranked or [], cfg
        )
        if activate_semantic:
            LOGGER.info("[isolate] semantic refinement activated")
            try:
                sam_alpha, sem_meta = apply_semantic_refinement(
                    rgb, alpha, labels, keep, cfg, stem=stem
                )
            except Exception as e:
                LOGGER.warning("[isolate] semantic refinement failed: %s", e)
                sam_alpha, sem_meta = None, {"reason": "exception", "error": str(e)}
            if sam_alpha is not None:
                masked_alpha = sam_alpha
                v3 = {**sem_meta, "used": True}
            else:
                LOGGER.info("[isolate] fallback to heuristic path")
                v3 = {**sem_meta, "used": False, "fallback_heuristic": True}
        elif cfg.semantic_refinement_enabled:
            v3["skipped"] = True
    elif cfg.semantic_refinement_enabled:
        v3["skipped"] = True
        v3["reason"] = "ranking_primary" if group_res is None else "grouping_primary"

    masked_alpha, bin_frag, clean_bin = strip_small_fragments(masked_alpha, cfg)
    masked_alpha = morphological_post_open(
        masked_alpha,
        cfg.morph_post_open_size,
        cfg.alpha_visibility_threshold,
    )

    masked_alpha = refine_alpha_soft(masked_alpha, cfg.edge_blur_sigma)

    LOGGER.info(
        "[debug] masked_alpha dtype=%s min=%s max=%s unique_sample=%s",
        masked_alpha.dtype,
        masked_alpha.min(),
        masked_alpha.max(),
        np.unique(masked_alpha)[:10],
    )
    if not np.any(masked_alpha > cfg.alpha_visibility_threshold):
        raise OSError("isolate produced empty mask after cleanup")

    out_rgba = compose_isolated_rgba(rgba, masked_alpha, cfg)

    write_isolate_debug(
        cfg,
        stem=stem,
        rgb=rgb,
        labels=labels,
        keep_label=keep,
        refined_alpha=masked_alpha,
        ranked=legacy_ranked,
    )

    fragments_delta = int(np.count_nonzero(bin_frag) - np.count_nonzero(clean_bin))
    LOGGER.info(
        "[isolate] cleaned artifacts fragment_delta~=%d sigma=%s",
        max(0, fragments_delta),
        cfg.edge_blur_sigma,
    )

    context.current_rgba = np.ascontiguousarray(out_rgba)
    context.alpha_mask = np.ascontiguousarray(masked_alpha)
    context.debug["isolate_component_count"] = n_fg
    context.debug["isolate_selected_label"] = keep

    if legacy_ranked is not None:
        best_feats = next(f for f in legacy_ranked if f.label == keep)
        sel_area = int(stats[keep, cv2.CC_STAT_AREA])
        context.debug["isolate_selection_scores"] = {
            int(f.label): round(float(f.confidence), 4) for f in legacy_ranked
        }
        context.debug["isolate_selected_area"] = sel_area
        context.debug["isolate_selected_confidence"] = round(float(best_feats.confidence), 4)
        context.debug["isolate_v2_ranked"] = [
            {
                "label": f.label,
                "area": f.area,
                "confidence": round(float(f.confidence), 4),
                "semantic": f.semantic,
                "relative_area": round(float(f.relative_area), 4),
                "border_contact_ratio": round(float(f.border_contact_ratio), 4),
                "solidity": round(float(f.solidity), 4),
                "elongation": round(float(f.elongation), 4),
                "complexity": round(float(f.complexity), 4),
                "bbox": [int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3])],
                "breakdown": {k: round(float(v), 4) for k, v in f.breakdown.items()},
            }
            for f in legacy_ranked
        ]
    elif rank_res is not None:
        context.debug["isolate_selection_scores"] = {}
        context.debug["isolate_selected_area"] = sel_area
        context.debug["isolate_selected_confidence"] = round(float(rank_res.ranked[0].confidence), 4)
        context.debug["isolate_v2_ranked"] = []

    context.debug["isolate_v3_semantic"] = v3
    if cfg.semantic_refinement_enabled:
        context.debug["semantic_activation_reason"] = activation_meta.get("reason") if activate_semantic else None
        context.debug["semantic_activation_detail"] = activation_meta

    LOGGER.info("[isolate] complete %s", name)
    return context
