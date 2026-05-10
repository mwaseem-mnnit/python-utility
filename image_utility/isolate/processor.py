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
    select_product_label,
)
from .config import IsolateConfig, load_isolate_config
from .debug import write_isolate_debug
from .refinement import compose_isolated_rgba, refine_alpha_soft
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

    keep, infos, scores = select_product_label(labels, stats, centroids, cfg)
    if keep is None:
        raise OSError("could not select a product component")

    LOGGER.info("[isolate] detected %d components", len(infos))

    sel_area = int(stats[keep, cv2.CC_STAT_AREA])
    LOGGER.info(
        "[isolate] selected label=%d area=%d",
        keep,
        sel_area,
    )

    # 4 Artifact cleanup
    masked_alpha = apply_kept_label_to_alpha(alpha, labels, keep)
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
    )

    fragments_delta = int(np.count_nonzero(bin_frag) - np.count_nonzero(clean_bin))
    LOGGER.info(
        "[isolate] cleaned artifacts fragment_delta~=%d sigma=%s",
        max(0, fragments_delta),
        cfg.edge_blur_sigma,
    )

    context.current_rgba = np.ascontiguousarray(out_rgba)
    context.alpha_mask = np.ascontiguousarray(masked_alpha)
    context.debug["isolate_component_count"] = len(infos)
    context.debug["isolate_selected_label"] = keep
    context.debug["isolate_selection_scores"] = {int(k): round(float(v), 4) for k, v in scores.items()}
    context.debug["isolate_selected_area"] = sel_area

    LOGGER.info("[isolate] complete %s", name)
    return context
