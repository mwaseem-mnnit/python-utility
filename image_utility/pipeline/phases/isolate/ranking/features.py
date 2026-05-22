"""Per-candidate feature extraction (ranking-owned; no cross-stage helpers)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import RankingConfig
from .contracts import FeatureVector, RankingMaskInput

BoolMask = NDArray[np.bool_]
UInt8 = NDArray[np.uint8]


def _border_contact_ratio(mask: BoolMask) -> float:
    area = int(np.count_nonzero(mask))
    if area < 1:
        return 0.0
    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] |= mask[0, :]
    edge[-1, :] |= mask[-1, :]
    edge[:, 0] |= mask[:, 0]
    edge[:, -1] |= mask[:, -1]
    return float(np.count_nonzero(np.logical_and(mask, edge))) / float(area)


def _mask_centroid(mask: BoolMask) -> tuple[float, float]:
    m = mask.astype(np.uint8)
    ys = m.sum(axis=1)
    xs = m.sum(axis=0)
    if ys.max() < 1 or xs.max() < 1:
        return 0.0, 0.0
    h, w = mask.shape
    cy = float((np.arange(h) * ys).sum() / ys.sum())
    cx = float((np.arange(w) * xs).sum() / xs.sum())
    return cx, cy


def _centroid_distance_norm(
    cx: float,
    cy: float,
    ih: int,
    iw: int,
    cfg: RankingConfig,
) -> float:
    cx0, cy0 = (iw - 1) * 0.5, (ih - 1) * 0.5
    diag = math.hypot(iw, ih)
    dist = math.hypot(cx - cx0, cy - cy0) / (0.5 * diag + cfg.math_epsilon)
    return float(dist)


def _solidity(mask_crop: UInt8) -> float:
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    if a < 1.0:
        return 1.0
    hull = cv2.convexHull(cnt)
    ha = float(cv2.contourArea(hull))
    return float(a / max(ha, 1.0))


def _contour_complexity(mask_u8: UInt8, bbox: tuple[int, int, int, int], cap: float) -> float:
    x, y, w, h = bbox
    crop = mask_u8[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    a = float(cv2.contourArea(cnt))
    if a < 1.0:
        return 0.0
    raw = float(peri / math.sqrt(a))
    return float(min(raw, cap))


def _overlap_with_fg(mask: BoolMask, fg: BoolMask) -> float:
    inter = int(np.count_nonzero(np.logical_and(mask, fg)))
    ma = int(np.count_nonzero(mask))
    return float(inter) / float(max(ma, 1))


def extract_features_for_candidate(
    prop: RankingMaskInput,
    *,
    fg_bool: BoolMask,
    total_fg_pixels: int,
    image_hw: tuple[int, int],
    cfg: RankingConfig,
) -> FeatureVector:
    """Pure extraction — does not rank or alter ``prop.mask``."""
    ih, iw = image_hw
    mask = prop.mask
    area = int(np.count_nonzero(mask))
    if area < 1:
        return FeatureVector(
            area=0,
            relative_area=0.0,
            centroid_xy=(0.0, 0.0),
            bbox_xywh=(0, 0, 0, 0),
            bbox_fill_ratio=0.0,
            solidity=1.0,
            elongation=1.0,
            border_contact_ratio=0.0,
            contour_complexity=0.0,
            occupancy_ratio=0.0,
            center_distance_norm=1.0,
            sam_predicted_iou=float(prop.predicted_iou),
            sam_stability=float(prop.stability_score),
            overlap_rembg_fg=0.0,
        )

    rel = float(area) / float(max(total_fg_pixels, 1))

    ys, xs = np.where(mask)
    if ys.size < 1:
        x0 = y0 = w = h = 0
    else:
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        w = x1 - x0 + 1
        h = y1 - y0 + 1
    bbox = (x0, y0, w, h)
    fill = float(area) / float(max(w * h, 1))

    cx, cy = _mask_centroid(mask)
    dist_norm = _centroid_distance_norm(cx, cy, ih, iw, cfg)

    u8 = (mask.astype(np.uint8) * 255).astype(np.uint8)
    crop = u8[y0 : y0 + h, x0 : x0 + w] if h > 0 and w > 0 else u8
    sol = _solidity(crop) if crop.size else 1.0
    elong = max(w, h) / max(min(w, h), 1)
    border = _border_contact_ratio(mask)
    complexity = _contour_complexity(u8, bbox, cfg.complexity_cap)
    occ = float(area) / float(max(ih * iw, 1))
    ov = _overlap_with_fg(mask, fg_bool)

    return FeatureVector(
        area=area,
        relative_area=rel,
        centroid_xy=(cx, cy),
        bbox_xywh=bbox,
        bbox_fill_ratio=float(min(fill, 1.0)),
        solidity=float(min(max(sol, 0.0), 1.0)),
        elongation=float(elong),
        border_contact_ratio=border,
        contour_complexity=complexity,
        occupancy_ratio=occ,
        center_distance_norm=dist_norm,
        sam_predicted_iou=float(prop.predicted_iou),
        sam_stability=float(prop.stability_score),
        overlap_rembg_fg=ov,
    )


def extract_all_features(
    proposals: tuple[RankingMaskInput, ...],
    base_alpha: UInt8,
    cfg: RankingConfig,
) -> tuple[FeatureVector, ...]:
    fg = base_alpha > cfg.alpha_visibility_threshold
    total_fg = int(np.count_nonzero(fg))
    if total_fg < 1:
        total_fg = 1
    ih, iw = base_alpha.shape[:2]
    return tuple(
        extract_features_for_candidate(
            p,
            fg_bool=fg,
            total_fg_pixels=total_fg,
            image_hw=(ih, iw),
            cfg=cfg,
        )
        for p in proposals
    )
