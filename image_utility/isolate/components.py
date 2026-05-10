"""Connected-component analysis, region statistics, and product selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

UInt8 = NDArray[np.uint8]
Labels = NDArray[np.int32]


@dataclass(frozen=True)
class ComponentInfo:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def binary_foreground_mask(alpha: UInt8, cfg: IsolateConfig) -> UInt8:
    """Binary uint8 mask (0 / 255) where alpha indicates foreground."""
    return np.where(alpha > cfg.alpha_visibility_threshold, 255, 0).astype(np.uint8)


def analyze_connected_components(mask_255: UInt8) -> tuple[Labels, np.ndarray, np.ndarray]:
    """
    Run connected components on a binary mask.

    Returns ``(labels, stats, centroids)`` (label 0 = background), OpenCV layout.
    """
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)
    _nlab, labels, stats, cents = cv2.connectedComponentsWithStats(mask_255)
    return labels, stats, cents


def _enumerate_components(stats: np.ndarray, centroids: np.ndarray) -> list[ComponentInfo]:
    infos: list[ComponentInfo] = []
    for i in range(1, stats.shape[0]):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        infos.append(ComponentInfo(label=i, area=area, bbox=(x, y, w, h), centroid=(cx, cy)))
    return infos


def _contour_complexity(mask_u8: UInt8, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    crop = mask_u8[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    best = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(best, True)
    a = float(cv2.contourArea(best))
    if a < 1.0:
        return 0.0
    return float(peri / math.sqrt(a))


def score_component(
    info: ComponentInfo,
    image_hw: tuple[int, int],
    labels: Labels,
    cfg: IsolateConfig,
) -> float:
    if info.area < cfg.min_component_area:
        return -1.0

    ih, iw = image_hw
    cx0, cy0 = (iw - 1) * 0.5, (ih - 1) * 0.5
    cx, cy = info.centroid
    diag = math.hypot(iw, ih)
    dist_norm = math.hypot(cx - cx0, cy - cy0) / (0.5 * diag + cfg.math_epsilon)
    center_factor = math.exp(-cfg.center_bias * dist_norm)

    _x, _y, bw, bh = info.bbox
    ar = max(bw, bh) / max(min(bw, bh), 1)
    elong = 1.0 if ar < cfg.aspect_ratio_penalty_threshold else cfg.elongation_penalty

    mask_bin = (labels == info.label).astype(np.uint8) * 255
    complexity = min(_contour_complexity(mask_bin, info.bbox), cfg.complexity_score_cap)

    area_factor = math.sqrt(float(info.area))
    structure = 1.0 + cfg.complexity_weight * complexity
    return float(area_factor * center_factor * structure * elong)


def select_product_label(
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: IsolateConfig,
) -> tuple[int | None, list[ComponentInfo], dict[int, float]]:
    """Pick the best foreground label; returns ``(label | None, infos, scores)``."""
    ih, iw = labels.shape[:2]
    infos = _enumerate_components(stats, centroids)
    scores: dict[int, float] = {}
    best_label: int | None = None
    best_score = -1.0
    for info in infos:
        s = score_component(info, (ih, iw), labels, cfg)
        scores[info.label] = s
        if s > best_score:
            best_score = s
            best_label = info.label
    if best_label is None or best_score <= 0:
        return None, infos, scores
    return best_label, infos, scores


def apply_kept_label_to_alpha(alpha: UInt8, labels: Labels, keep_label: int) -> UInt8:
    """Zero alpha outside the kept connected-component region (preserve soft alpha inside)."""
    mask = labels == keep_label
    out = alpha.copy()
    out[~mask] = 0
    return out
