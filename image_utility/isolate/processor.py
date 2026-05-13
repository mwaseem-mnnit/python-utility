"""Isolate pipeline orchestration (high-level steps only)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext

from .cleanup import (
    morphological_post_open,
    morphological_pre_cc,
    strip_small_fragments,
)
from .components import (
    analyze_connected_components,
    apply_kept_label_to_alpha,
    binary_foreground_mask,
    select_best_component,
)
from .config import IsolateConfig, load_isolate_config
from .debug import write_isolate_debug
from .refinement import compose_isolated_rgba, refine_alpha_soft
from .semantic.refinement import apply_semantic_refinement, should_activate_semantic_refinement
from .segmentation import extract_alpha, segment_rgba

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

    # 1–2 Segmentation + alpha
    rgba = segment_rgba(rgb, cfg)
    alpha = extract_alpha(rgba)
    LOGGER.info("[isolate] segmented %s", name)

    if not np.any(alpha > cfg.alpha_visibility_threshold):
        raise OSError("segmentation collapse: empty alpha")

    # 3 Component analysis (pre-mask + CC + selection)
    bin_m = binary_foreground_mask(alpha, cfg)
    bin_m = morphological_pre_cc(bin_m, cfg.morph_pre_close_size)
    labels, stats, centroids = analyze_connected_components(bin_m)
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

    # 4 Artifact cleanup (heuristic CC mask, optionally replaced by SAM v3)
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

    # 5 Edge refinement + RGBA
    masked_alpha = refine_alpha_soft(masked_alpha, cfg.edge_blur_sigma)

    if not np.any(masked_alpha > cfg.alpha_visibility_threshold):
        raise OSError("isolate produced empty mask after cleanup")

    out_rgba = compose_isolated_rgba(rgba, masked_alpha, cfg)

    # 6 Debug output
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
