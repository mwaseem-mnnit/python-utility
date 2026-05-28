"""Isolate pipeline orchestration (high-level steps only)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext
from image_utility.pipeline.phases.isolate.decomposition import DecompositionProcessor, DecompositionResult
from image_utility.pipeline.phases.isolate.filtering import (
    FilteringInput,
    FilteringProcessor,
    FilteringProposal,
    load_stop_after_aliases,
)
from image_utility.pipeline.phases.isolate.grouping import (
    GroupingCandidateInput,
    GroupingFeatureVector,
    GroupingInput,
    GroupingProcessor,
    GroupingRankingSnapshot,
    GroupingResult,
)
from image_utility.pipeline.phases.isolate.ranking import (
    RankingInput,
    RankingMaskInput,
    RankingProcessor,
    RankingResult,
)
from image_utility.pipeline.phases.isolate.ownership import (
    OwnedGroup,
    OwnershipGroupedRegionInput,
    OwnershipGroupingSnapshot,
    OwnershipInput,
    OwnershipProcessor,
    OwnershipRankingSnapshot,
    OwnershipResult,
)
from image_utility.pipeline.phases.isolate.suppression import (
    SuppressionGroupedRegionInput,
    SuppressionGroupingSnapshot,
    SuppressionInput,
    SuppressionProcessor,
    SuppressionRankingSnapshot,
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


def _build_filtering_input(decomp: DecompositionResult) -> FilteringInput:
    """Map decomposition proposals into filtering DTOs only (no decomposition imports in filtering)."""

    props: list[FilteringProposal] = []
    for c in decomp.semantic_candidates:
        ar = int(c.area) if int(c.area) > 0 else int(np.count_nonzero(c.mask))
        props.append(
            FilteringProposal(
                candidate_id=int(c.candidate_id),
                mask=np.ascontiguousarray(c.mask),
                source=str(c.source),
                predicted_iou=float(c.predicted_iou),
                stability_score=float(c.stability_score),
                area=ar,
            )
        )
    hh, ww = decomp.base_alpha.shape[:2]
    return FilteringInput(proposals=tuple(props), image_hw=(hh, ww))


def _build_ownership_input(group_res: GroupingResult, rank_res: RankingResult | None) -> OwnershipInput:
    """Copy grouping output into ownership contracts (no ownership→grouping imports)."""
    regions: list[OwnershipGroupedRegionInput] = []
    for g in group_res.groups:
        regions.append(
            OwnershipGroupedRegionInput(
                group_id=g.group_id,
                member_candidate_ids=g.member_candidate_ids,
                grouped_mask=np.ascontiguousarray(g.grouped_mask),
                group_confidence=float(g.group_confidence),
                affinity_breakdown=dict(g.affinity_breakdown),
                geometry_metadata=dict(g.geometry_metadata),
            )
        )
    hh, ww = regions[0].grouped_mask.shape[:2]
    rk = rank_res.metadata if rank_res is not None else None
    ranking_snap = (
        OwnershipRankingSnapshot(
            candidate_count=rk.candidate_count,
            ambiguity_detected=rk.ambiguity_detected,
            top_confidence=float(rk.top_confidence),
            second_confidence=float(rk.second_confidence),
            confidence_separation=float(rk.confidence_separation),
        )
        if rk is not None
        else None
    )
    gm = group_res.metadata
    grouping_snap = OwnershipGroupingSnapshot(
        group_count=gm.group_count,
        grouping_ambiguity=gm.grouping_ambiguity,
        top_group_confidence=float(gm.top_group_confidence),
        second_group_confidence=float(gm.second_group_confidence),
    )
    return OwnershipInput(
        regions=tuple(regions),
        image_hw=(hh, ww),
        ranking_snapshot=ranking_snap,
        grouping_snapshot=grouping_snap,
    )


def _build_suppression_input_from_ownership(
    own_res: OwnershipResult,
    rank_res: RankingResult | None,
    group_res: GroupingResult,
) -> SuppressionInput:
    """
    Flatten ownership-labelled groups into suppression contracts.

    The ownership label and confidence are stored in affinity_breakdown so that
    suppression can treat them as additional evidence without importing ownership types.
    """
    regions: list[SuppressionGroupedRegionInput] = []
    for g in own_res.owned_groups:
        # Forward ownership scores into affinity_breakdown for suppression visibility
        aff = dict(g.ownership_breakdown)
        regions.append(
            SuppressionGroupedRegionInput(
                group_id=g.group_id,
                member_candidate_ids=g.member_candidate_ids,
                grouped_mask=np.ascontiguousarray(g.surviving_mask),  # already clipped
                group_confidence=float(g.ownership_confidence),
                affinity_breakdown=aff,
                geometry_metadata=dict(g.geometry_metadata),
            )
        )
    hh, ww = regions[0].grouped_mask.shape[:2]
    rk = rank_res.metadata if rank_res is not None else None
    ranking_snap = (
        SuppressionRankingSnapshot(
            candidate_count=rk.candidate_count,
            ambiguity_detected=rk.ambiguity_detected,
            top_confidence=float(rk.top_confidence),
            second_confidence=float(rk.second_confidence),
            confidence_separation=float(rk.confidence_separation),
        )
        if rk is not None
        else None
    )
    gm = group_res.metadata
    grouping_snap = SuppressionGroupingSnapshot(
        group_count=gm.group_count,
        grouping_ambiguity=gm.grouping_ambiguity,
        top_group_confidence=float(own_res.metadata.global_ownership_confidence),
        second_group_confidence=float(gm.second_group_confidence),
    )
    return SuppressionInput(
        regions=tuple(regions),
        image_hw=(hh, ww),
        ranking_snapshot=ranking_snap,
        grouping_snapshot=grouping_snap,
    )


def _build_suppression_input(group_res: GroupingResult, rank_res: RankingResult | None) -> SuppressionInput:
    """Flatten grouping output directly into suppression contracts (ownership-bypass path)."""
    regions: list[SuppressionGroupedRegionInput] = []
    for g in group_res.groups:
        regions.append(
            SuppressionGroupedRegionInput(
                group_id=g.group_id,
                member_candidate_ids=g.member_candidate_ids,
                grouped_mask=g.grouped_mask,
                group_confidence=float(g.group_confidence),
                affinity_breakdown=dict(g.affinity_breakdown),
                geometry_metadata=dict(g.geometry_metadata),
            )
        )
    hh, ww = regions[0].grouped_mask.shape[:2]
    rk = rank_res.metadata if rank_res is not None else None
    ranking_snap = (
        SuppressionRankingSnapshot(
            candidate_count=rk.candidate_count,
            ambiguity_detected=rk.ambiguity_detected,
            top_confidence=float(rk.top_confidence),
            second_confidence=float(rk.second_confidence),
            confidence_separation=float(rk.confidence_separation),
        )
        if rk is not None
        else None
    )
    gm = group_res.metadata
    grouping_snap = SuppressionGroupingSnapshot(
        group_count=gm.group_count,
        grouping_ambiguity=gm.grouping_ambiguity,
        top_group_confidence=float(gm.top_group_confidence),
        second_group_confidence=float(gm.second_group_confidence),
    )
    return SuppressionInput(
        regions=tuple(regions),
        image_hw=(hh, ww),
        ranking_snapshot=ranking_snap,
        grouping_snapshot=grouping_snap,
    )


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

    stop_after = load_stop_after_aliases()
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

    filter_fallback = False
    filter_res = None
    try:
        f_in = _build_filtering_input(decomp)
        filter_res = FilteringProcessor().run(f_in, rgb, stem)
    except Exception as exc:
        LOGGER.warning("[isolate] filtering stage failed (%s); using raw decomposition proposals", exc)
        filter_fallback = True

    if filter_res is not None:
        context.metadata["filtering_result"] = filter_res
        context.debug["filtering"] = {
            "input_count": filter_res.metadata.input_count,
            "accepted_count": filter_res.metadata.accepted_count,
            "rejected_count": filter_res.metadata.rejected_count,
            "all_rejected_fallback": filter_res.metadata.all_rejected_fallback,
        }

    if stop_after == "filtering":
        if filter_fallback or filter_res is None:
            raise OSError(
                "stop-after-filtering requires a successful filtering run "
                "(set IMAGE_UTIL_ISOLATE_STOP_AFTER_STAGE=filtering or ISOLATE_STOP_AFTER_STAGE=filtering)"
            )
        sem = np.zeros_like(alpha, dtype=bool)
        for p in filter_res.accepted:
            sem = np.logical_or(sem, p.mask)
        if not np.any(sem):
            raise OSError("filtering produced empty semantic union — cannot stop after filtering")
        stop_alpha = np.where(sem, 255, 0).astype(np.uint8)
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(stop_alpha)
        context.metadata["isolate_stopped_after"] = "filtering"
        context.debug["isolate_stopped_after"] = "filtering"
        LOGGER.info("[isolate] stop after filtering — ranking/grouping/suppression/refinement skipped")
        LOGGER.info("[isolate] complete %s (filtering-only)", name)
        return context

    if filter_fallback or filter_res is None:
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
    else:
        proposals = tuple(
            RankingMaskInput(
                candidate_id=p.candidate_id,
                mask=p.mask,
                source=p.source,
                predicted_iou=p.predicted_iou,
                stability_score=p.stability_score,
                area=p.area,
            )
            for p in filter_res.accepted
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
        LOGGER.info("[isolate] stop after grouping — ownership/suppression/refinement skipped")
        LOGGER.info("[isolate] complete %s (grouping-only)", name)
        return context

    # ── Ownership stage (grouping → ownership → suppression) ─────────────────
    own_res = None
    ownership_fallback = False
    if group_res is not None and group_res.groups:
        try:
            own_in = _build_ownership_input(group_res, rank_res)
            own_res = OwnershipProcessor().run(own_in, rgb, stem)
        except Exception as exc:
            LOGGER.warning(
                "[isolate] ownership stage failed (%s); passing grouping output directly to suppression",
                exc,
            )
            ownership_fallback = True
    else:
        ownership_fallback = True

    if own_res is not None:
        context.metadata["ownership_result"] = own_res
        context.debug["ownership"] = {
            "product_group_count": own_res.metadata.product_group_count,
            "support_group_count": own_res.metadata.support_group_count,
            "packaging_group_count": own_res.metadata.packaging_group_count,
            "environment_group_count": own_res.metadata.environment_group_count,
            "uncertain_group_count": own_res.metadata.uncertain_group_count,
            "support_clipped_group_count": own_res.metadata.support_clipped_group_count,
            "global_ownership_confidence": round(own_res.metadata.global_ownership_confidence, 4),
            "primary_product_group_id": own_res.metadata.primary_product_group_id,
        }
        context.debug["isolate_ownership_groups"] = [
            {
                "group_id": g.group_id,
                "ownership_label": g.ownership_label,
                "ownership_confidence": round(g.ownership_confidence, 4),
                "product_likelihood": round(g.product_likelihood, 4),
                "support_likelihood": round(g.support_likelihood, 4),
                "removed_support_px": int(np.count_nonzero(g.removed_support_mask)),
            }
            for g in own_res.owned_groups[:12]
        ]

    if stop_after == "ownership":
        need_prev = (
            ranking_fallback or rank_res is None
            or group_res is None or grouping_fallback
            or own_res is None or ownership_fallback
        )
        if need_prev:
            raise OSError(
                "ISOLATE_STOP_AFTER_STAGE=ownership requires successful ranking, grouping, "
                "and ownership runs"
            )
        semantic = own_res.combined_product_mask
        if not np.any(semantic):
            raise OSError("ownership produced empty product mask — cannot stop after ownership")
        stop_alpha = np.where(semantic, 255, 0).astype(np.uint8)
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(stop_alpha)
        context.metadata["isolate_stopped_after"] = "ownership"
        context.debug["isolate_stopped_after"] = "ownership"
        LOGGER.info("[isolate] stop after ownership — suppression/refinement skipped")
        LOGGER.info("[isolate] complete %s (ownership-only)", name)
        return context

    suppress_res = None
    suppression_fallback = False
    if own_res is not None and own_res.owned_groups:
        try:
            suppress_in = _build_suppression_input_from_ownership(own_res, rank_res, group_res)
            suppress_res = SuppressionProcessor().run(suppress_in, rgb, stem)
        except Exception as exc:
            LOGGER.warning(
                "[isolate] suppression stage failed (%s); falling back without semantic cleanup",
                exc,
            )
            suppression_fallback = True
    elif group_res is not None and group_res.groups:
        # Ownership failed — feed grouping output directly to suppression
        try:
            suppress_in = _build_suppression_input(group_res, rank_res)
            suppress_res = SuppressionProcessor().run(suppress_in, rgb, stem)
        except Exception as exc:
            LOGGER.warning(
                "[isolate] suppression stage failed (%s); falling back without semantic cleanup",
                exc,
            )
            suppression_fallback = True
    else:
        suppression_fallback = True

    if suppress_res is not None:
        context.metadata["suppression_result"] = suppress_res
        context.debug["suppression"] = {
            "removed_group_ids": list(suppress_res.metadata.removed_group_ids),
            "surviving_group_count": suppress_res.metadata.surviving_group_count,
            "removed_group_count": suppress_res.metadata.removed_group_count,
            "global_suppression_confidence": round(
                suppress_res.metadata.global_suppression_confidence,
                4,
            ),
        }
        context.debug["isolate_suppression_top"] = [
            {
                "group_id": g.group_id,
                "member_candidate_ids": list(g.member_candidate_ids),
                "suppression_confidence": round(g.suppression_confidence, 4),
                "removed_region_ids": list(g.removed_region_ids),
            }
            for g in suppress_res.surviving_groups[:12]
        ]

    if stop_after == "suppression":
        need_group = ranking_fallback or rank_res is None or group_res is None or grouping_fallback
        if need_group or suppress_res is None or suppression_fallback:
            raise OSError(
                "ISOLATE_STOP_AFTER_STAGE=suppression requires successful ranking, grouping, "
                "and suppression runs"
            )
        semantic = suppress_res.combined_survivor_mask
        if not np.any(semantic):
            raise OSError("suppression produced empty survivor mask — cannot stop after suppression")
        stop_alpha = np.where(semantic, 255, 0).astype(np.uint8)
        context.current_rgba = np.ascontiguousarray(decomp.base_rgba)
        context.alpha_mask = np.ascontiguousarray(stop_alpha)
        context.metadata["isolate_stopped_after"] = "suppression"
        context.debug["isolate_stopped_after"] = "suppression"
        LOGGER.info("[isolate] stop after suppression — refinement skipped")
        LOGGER.info("[isolate] complete %s (suppression-only)", name)
        return context

    legacy_ranked = None
    keep = 1
    if suppress_res is not None and np.any(suppress_res.combined_survivor_mask):
        semantic = suppress_res.combined_survivor_mask
        masked_alpha = np.where(semantic, 255, 0).astype(np.uint8)
        keep = _cc_label_best_overlap(labels, stats, semantic)
    elif group_res is not None and group_res.groups:
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
        masked_alpha = np.where(top.mask, 255, 0).astype(np.uint8)
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
    elif suppress_res is not None and suppress_res.surviving_groups:
        sg = suppress_res.surviving_groups[0]
        LOGGER.info(
            "[isolate] suppression-primary groups=%d top_group_id=%d confidence=%.2f meta=%.2f",
            len(suppress_res.surviving_groups),
            sg.group_id,
            sg.suppression_confidence,
            suppress_res.metadata.global_suppression_confidence,
        )
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
        if suppress_res is not None and np.any(suppress_res.combined_survivor_mask):
            v3["reason"] = "suppression_primary"
        elif group_res is not None and group_res.groups:
            v3["reason"] = "grouping_primary"
        else:
            v3["reason"] = "ranking_primary"

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
    elif suppress_res is not None and suppress_res.surviving_groups:
        context.debug["isolate_selection_scores"] = {}
        context.debug["isolate_selected_area"] = sel_area
        context.debug["isolate_selected_confidence"] = round(
            float(suppress_res.metadata.global_suppression_confidence), 4
        )
        context.debug["isolate_v2_ranked"] = []
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
