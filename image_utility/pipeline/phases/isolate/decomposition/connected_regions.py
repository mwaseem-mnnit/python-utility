"""Connected-component topology from alpha (metadata only; no selection)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import DecompositionConfig
from .contracts import ConnectedRegion, Labels, UInt8

UInt8RGB = NDArray[np.uint8]


def binary_from_alpha(alpha: UInt8, thresh: int) -> UInt8:
    return np.where(alpha > thresh, 255, 0).astype(np.uint8)


def morph_pre_close(mask_255: UInt8, ksize: int) -> UInt8:
    if ksize <= 0:
        return mask_255
    k = ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, ker)


def analyze_components(mask_255: UInt8) -> tuple[Labels, np.ndarray, np.ndarray]:
    if mask_255.dtype != np.uint8:
        mask_255 = mask_255.astype(np.uint8)
    _nlab, labels, stats, cents = cv2.connectedComponentsWithStats(mask_255)
    return labels, stats, cents


def _solidity_for_label(labels: Labels, label_id: int, bbox: tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    m = (labels == label_id).astype(np.uint8)[y : y + h, x : x + w]
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    cnt = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(cnt))
    if a < 1.0:
        return 1.0
    hull = cv2.convexHull(cnt)
    ha = float(cv2.contourArea(hull))
    return float(a / max(ha, 1.0))


def extract_connected_regions(
    labels: Labels,
    stats: np.ndarray,
    centroids: np.ndarray,
    cfg: DecompositionConfig,
) -> tuple[ConnectedRegion, ...]:
    """Build per-component metadata including contour counts (no suppression beyond tiny dust)."""
    n = stats.shape[0]
    if n <= 1:
        return tuple()

    out: list[ConnectedRegion] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_region_area_cc:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        bbox = (x, y, w, h)
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        m = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pt_count = sum(int(c.shape[0]) for c in contours)
        sol = _solidity_for_label(labels, i, bbox)
        out.append(
            ConnectedRegion(
                label=i,
                area=area,
                bbox=bbox,
                centroid_xy=(cx, cy),
                contour_point_count=pt_count,
                solidity=float(min(max(sol, 0.0), 1.0)),
            )
        )
    return tuple(out)
