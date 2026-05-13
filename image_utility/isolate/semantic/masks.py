"""Prepare SAM segmentations constrained to rembg foreground (conservative)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ..config import IsolateConfig

BoolMask = NDArray[np.bool_]


@dataclass
class PreparedMask:
    """One candidate region after intersecting SAM with rembg foreground."""

    mask: BoolMask
    overlap_rembg: float
    pred_iou: float
    stability: float
    raw_area: int


def foreground_bool(alpha: NDArray[np.uint8], thresh: int) -> BoolMask:
    return alpha > thresh


def prepare_sam_candidates(
    raw_masks: list[dict],
    fg_bool: BoolMask,
    cfg: IsolateConfig,
) -> list[PreparedMask]:
    """
    Intersect each SAM mask with ``fg_bool``, drop tiny islands, keep overlap metadata.

    ``overlap_rembg`` is |mask ∩ fg| / |mask| on the **original** SAM polygon before
    hard intersection (high → mask agrees with rembg silhouette).
    """
    out: list[PreparedMask] = []
    fg_area = int(np.count_nonzero(fg_bool))
    if fg_area < 1:
        return out

    for entry in raw_masks:
        seg = entry.get("segmentation")
        if seg is None:
            continue
        raw = np.asarray(seg, dtype=bool)
        if raw.shape != fg_bool.shape:
            raw = cv2.resize(raw.astype(np.uint8), (fg_bool.shape[1], fg_bool.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        raw_area = int(np.count_nonzero(raw))
        if raw_area < cfg.semantic_sam_min_mask_area:
            continue

        inter = np.logical_and(raw, fg_bool)
        inter_area = int(np.count_nonzero(inter))
        overlap = float(inter_area) / float(max(raw_area, 1))

        restricted = np.logical_and(raw, fg_bool)
        area_r = int(np.count_nonzero(restricted))
        if area_r < max(cfg.min_component_area, cfg.semantic_sam_min_mask_area // 2):
            continue
        if area_r / fg_area < cfg.semantic_candidate_min_fg_ratio:
            continue

        pred_iou = float(entry.get("predicted_iou", 0.0))
        stability = float(entry.get("stability_score", 0.0))

        out.append(
            PreparedMask(
                mask=restricted,
                overlap_rembg=overlap,
                pred_iou=pred_iou,
                stability=stability,
                raw_area=raw_area,
            )
        )
    return out
