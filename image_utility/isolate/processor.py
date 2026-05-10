"""Isolate phase orchestration (delegates to segmentation, cleanup, debug)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from image_utility.pipeline.context import PipelineContext

from .cleanup import (
    analyze_components,
    apply_label_mask_to_alpha,
    binary_foreground_mask,
    compose_isolated_rgba,
    morphological_post_open,
    morphological_pre_cc,
    refine_alpha_soft,
    remove_tiny_islands,
    select_product_label,
)
from .config import IsolateConfig, load_isolate_config
from .debug import save_alpha_png, save_components_viz, save_selection_overlay
from .segmentation import alpha_channel, rgba_from_rgb

LOGGER = logging.getLogger(__name__)


def process_isolate(context: PipelineContext, *, cfg: IsolateConfig | None = None) -> PipelineContext:
    """
    Populates ``context.current_rgba`` and ``context.alpha_mask`` (H×W uint8).

    Raises ``OSError`` on recoverable failures so the runner can skip the file.
    """
    cfg = cfg or load_isolate_config()
    rgb = context.current_image
    if rgb is None:
        raise OSError("isolate requires current_image (RGB)")

    stem = context.input_path.stem
    LOGGER.info("Isolate: segmentation start %s", context.input_path.name)

    rgba = rgba_from_rgb(rgb, model_name=cfg.rembg_model_name)
    alpha = alpha_channel(rgba)

    if not np.any(alpha > cfg.alpha_visibility_threshold):
        raise OSError("segmentation collapse: empty alpha")

    bin_m = binary_foreground_mask(alpha, cfg)
    bin_m = morphological_pre_cc(bin_m, cfg.morph_pre_close_size)

    labels, stats, centroids = analyze_components(bin_m)
    if stats.shape[0] <= 1:
        raise OSError("no foreground components detected")

    keep, infos, scores = select_product_label(labels, stats, centroids, cfg)
    if keep is None:
        raise OSError("could not select a product component")

    sel_area = int(stats[keep, cv2.CC_STAT_AREA])
    LOGGER.info(
        "Isolate: components=%d selected_label=%d area=%d",
        len(infos),
        keep,
        sel_area,
    )

    masked_alpha = apply_label_mask_to_alpha(alpha, labels, keep)

    _, bin_frag = cv2.threshold(
        masked_alpha, cfg.alpha_visibility_threshold, 255, cv2.THRESH_BINARY
    )
    clean_bin = remove_tiny_islands(bin_frag, min_area=cfg.min_fragment_area_after_select)
    masked_alpha = np.where(clean_bin > 0, masked_alpha, 0).astype(np.uint8)

    masked_alpha = morphological_post_open(
        masked_alpha, cfg.morph_post_open_size, cfg.alpha_visibility_threshold
    )
    masked_alpha = refine_alpha_soft(masked_alpha, cfg.edge_blur_sigma)

    if not np.any(masked_alpha > cfg.alpha_visibility_threshold):
        raise OSError("isolate produced empty mask after cleanup")

    out_rgba = compose_isolated_rgba(rgba, masked_alpha, cfg)

    if cfg.debug_enabled:
        save_alpha_png(stem, masked_alpha)
        save_components_viz(stem, labels)
        save_selection_overlay(stem, rgb, labels, keep)

    fragments_removed = int(np.count_nonzero(bin_frag) - np.count_nonzero(clean_bin))
    LOGGER.info(
        "Isolate: cleanup fragment_pixels~=%d edge_sigma=%s",
        max(0, fragments_removed),
        cfg.edge_blur_sigma,
    )

    context.current_rgba = np.ascontiguousarray(out_rgba)
    context.alpha_mask = np.ascontiguousarray(masked_alpha)
    context.debug["isolate_component_count"] = len(infos)
    context.debug["isolate_selected_label"] = keep
    context.debug["isolate_selection_scores"] = {int(k): round(float(v), 4) for k, v in scores.items()}
    context.debug["isolate_selected_area"] = sel_area

    LOGGER.info("Isolate: complete %s", context.input_path.name)
    return context
