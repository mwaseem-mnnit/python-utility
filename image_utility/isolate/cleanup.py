"""Connected components, product selection, artifact cleanup, alpha refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import IsolateConfig


@dataclass(frozen=True)
class ComponentInfo:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def binary_foreground_mask(alpha: np.ndarray, cfg: IsolateConfig) -> np.ndarray:
    """Binary uint8 mask (0 / 255) where alpha indicates foreground."""
    return np.where(alpha > cfg.alpha_visibility_threshold, 255, 0).astype(np.uint8)


def analyze_components(mask_255: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run connected components on a binary mask.

    Returns ``(labels, stats, centroids)`` as from OpenCV (label 0 = background).
    """
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)
    nlab, labels, stats, cents = cv2.connectedComponentsWithStats(mask_255)
    if nlab <= 1:
        return labels, stats, cents
    return labels, stats, cents


def _component_infos(
    labels: np.ndarray, stats: np.ndarray, centroids: np.ndarray
) -> list[ComponentInfo]:
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


def _contour_complexity(mask_region: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    crop = mask_region[y : y + h, x : x + w]
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
    shape: tuple[int, int],
    labels: np.ndarray,
    cfg: IsolateConfig,
) -> float:
    if info.area < cfg.min_component_area:
        return -1.0

    ih, iw = shape
    cx0, cy0 = (iw - 1) * 0.5, (ih - 1) * 0.5
    cx, cy = info.centroid
    diag = math.hypot(iw, ih)
    dist_norm = math.hypot(cx - cx0, cy - cy0) / (0.5 * diag + 1e-6)
    center_factor = math.exp(-cfg.center_bias * dist_norm)

    _, _, bw, bh = info.bbox
    ar = max(bw, bh) / max(min(bw, bh), 1)
    elong = 1.0 if ar < cfg.aspect_ratio_penalty_threshold else cfg.elongation_penalty

    mask_bin = (labels == info.label).astype(np.uint8) * 255
    complexity = min(_contour_complexity(mask_bin, info.bbox), 10.0)

    area_factor = math.sqrt(float(info.area))
    structure = 1.0 + cfg.complexity_weight * complexity
    return float(area_factor * center_factor * structure * elong)


def select_product_label(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: IsolateConfig,
) -> tuple[int | None, list[ComponentInfo], dict[int, float]]:
    """Pick the best foreground label; returns ``(label | None, infos, scores)``."""
    ih, iw = labels.shape[:2]
    infos = _component_infos(labels, stats, centroids)
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


def apply_label_mask_to_alpha(alpha: np.ndarray, labels: np.ndarray, keep: int) -> np.ndarray:
    """Zero alpha outside the kept label (preserve soft alpha inside)."""
    m = labels == keep
    out = alpha.copy()
    out[~m] = 0
    return out


def remove_tiny_islands(
    binary_mask: np.ndarray,
    *,
    min_area: int,
) -> np.ndarray:
    """Remove small connected components (4-connectivity) from binary mask."""
    lab, lbls, st, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=4)
    if lab <= 1:
        return binary_mask
    keep = np.ones(lab, dtype=bool)
    keep[0] = False
    for i in range(1, lab):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            keep[i] = False
    return np.isin(lbls, np.flatnonzero(keep)).astype(np.uint8) * 255


def morphological_pre_cc(mask_255: np.ndarray, close_ksize: int) -> np.ndarray:
    """Small closing before CC to merge near-touching product alpha."""
    if close_ksize <= 0:
        return mask_255
    k = close_ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, ker)


def morphological_post_open(alpha: np.ndarray, open_ksize: int, bin_thresh: int) -> np.ndarray:
    """Light opening on binarized alpha to shed thin bridges (optional)."""
    if open_ksize <= 0:
        return alpha
    k = open_ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    _, bin_m = cv2.threshold(alpha, bin_thresh, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, ker)
    return np.where(opened > 0, alpha, 0).astype(np.uint8)


def refine_alpha_soft(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Mild Gaussian smoothing on alpha (feather edges)."""
    if sigma <= 0:
        return alpha
    a = alpha.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def compose_isolated_rgba(original_rgba: np.ndarray, new_alpha: np.ndarray, cfg: IsolateConfig) -> np.ndarray:
    """Build RGBA with cleaned alpha; zero RGB where alpha is effectively zero."""
    out = np.zeros_like(original_rgba)
    out[:, :, 3] = new_alpha
    rgb = original_rgba[:, :, :3]
    visible = new_alpha[:, :, np.newaxis] > cfg.rgb_zero_below_alpha
    out[:, :, :3] = np.where(visible, rgb, 0).astype(np.uint8)
    return out
