"""Isolate pipeline orchestration (high-level steps only)."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext
from image_utility.pipeline.phases.isolate.decomposition import DecompositionProcessor

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

    # --- Decomposition stage (semantic proposals; bridge for legacy path) ---
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

    keep, ranked = select_best_component(labels, stats, centroids, cfg)
    if keep is None:
        raise OSError("could not select a product component")

    n_fg = int(stats.shape[0] - 1)
    LOGGER.info("[isolate] ranked %d foreground components", n_fg)

    sel_area = int(stats[keep, cv2.CC_STAT_AREA])
    best_feats = next(f for f in ranked if f.label == keep)
    LOGGER.info("[isolate] selected component confidence=%.2f", best_feats.confidence)
    LOGGER.info("[isolate] selected label=%d area=%d", keep, sel_area)
    if cfg.v2_weight_border_contact > 0 and best_feats.border_contact_ratio > 0.02:
        LOGGER.info("[isolate] applied border contact penalty")

    # Legacy refinement path (optional SAM v3) — to be superseded by isolate ranking/grouping stages
    masked_alpha = apply_kept_label_to_alpha(alpha, labels, keep)
    v3: dict[str, object] = {"used": False}
    activate_semantic, activation_meta = should_activate_semantic_refinement(stats, alpha, ranked, cfg)

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

    masked_alpha, bin_frag, clean_bin = strip_small_fragments(masked_alpha, cfg)
    masked_alpha = morphological_post_open(
        masked_alpha,
        cfg.morph_post_open_size,
        cfg.alpha_visibility_threshold,
    )

    masked_alpha = refine_alpha_soft(masked_alpha, cfg.edge_blur_sigma)

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
        ranked=ranked,
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
    context.debug["isolate_selection_scores"] = {
        int(f.label): round(float(f.confidence), 4) for f in ranked
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
        for f in ranked
    ]
    context.debug["isolate_v3_semantic"] = v3
    if cfg.semantic_refinement_enabled:
        context.debug["semantic_activation_reason"] = activation_meta.get("reason") if activate_semantic else None
        context.debug["semantic_activation_detail"] = activation_meta

    LOGGER.info("[isolate] complete %s", name)
    return context
