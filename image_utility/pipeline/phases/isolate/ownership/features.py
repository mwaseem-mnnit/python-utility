"""Ownership feature extraction — deterministic geometry descriptors."""
from __future__ import annotations
import math
import cv2
import numpy as np
from numpy.typing import NDArray
from .config import OwnershipConfig
from .contracts import OwnershipFeatures, OwnershipGroupedRegionInput
BoolMask = NDArray[np.bool_]
def _border_contact_ratio(mask: BoolMask, eps: float) -> float:
    h, w = mask.shape[:2]
    if h < 2 or w < 2 or not np.any(mask):
        return 0.0
    border = np.zeros_like(mask, dtype=np.uint8)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1
    inter = np.logical_and(mask, border.astype(bool))
    fg = int(np.count_nonzero(mask))
    return float(np.count_nonzero(inter) / max(fg, eps))
def _contour_solidity_complexity(mask: BoolMask) -> tuple[float, float]:
    """Returns (solidity, contour_complexity). complexity = perimeter^2/(4*pi*area)."""
    m8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0, 1.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area < 1.0:
        return 1.0, 1.0
    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / max(hull_area, 1.0))
    perimeter = float(cv2.arcLength(largest, True))
    complexity = float((perimeter ** 2) / max(4.0 * math.pi * area, 1.0))
    return float(np.clip(solidity, 0.0, 1.0)), float(max(1.0, complexity))
def _irregular_boundary_score(mask: BoolMask) -> float:
    """(convex_hull_area - pixel_area) / convex_hull_area — 0=compact; 1=very irregular."""
    m8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    hull_area = float(cv2.contourArea(cv2.convexHull(largest)))
    if hull_area < 1.0:
        return 0.0
    return float(np.clip((hull_area - area) / hull_area, 0.0, 1.0))
def _multi_blob_stats(mask: BoolMask, min_area: int = 10) -> tuple[int, float, float, float]:
    """Returns (blob_count, primary_coverage, secondary_ratio, finger_like_ratio)."""
    m8 = (mask.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    total_area = int(np.count_nonzero(mask))
    if total_area == 0 or n_labels < 2:
        return 1, 1.0, 0.0, 0.0
    blob_areas = sorted(
        [int(stats[lab, cv2.CC_STAT_AREA]) for lab in range(1, n_labels)
         if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area],
        reverse=True,
    )
    if not blob_areas:
        return 1, 1.0, 0.0, 0.0
    primary = blob_areas[0]
    primary_coverage = float(primary / max(total_area, 1))
    secondary_ratio = float(blob_areas[1] / max(primary, 1)) if len(blob_areas) > 1 else 0.0
    # Finger-like sub-blobs: elongated + thin
    finger_count = 0
    for lab in range(1, n_labels):
        a = int(stats[lab, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        elong = float(max(bw, bh) / max(min(bw, bh), 1))
        if elong >= 2.5:
            finger_count += 1
    finger_like_ratio = float(finger_count / max(len(blob_areas), 1))
    return len(blob_areas), primary_coverage, secondary_ratio, finger_like_ratio
def _thin_bridge_score(
    mask: BoolMask,
    image_hw: tuple[int, int],
    cfg: OwnershipConfig,
) -> tuple[float, int]:
    """
    Probe thin bridges by multi-scale erosion.
    Returns (bridge_score 0-1, smallest_radius_that_caused_split).
    """
    h, w = image_hw
    cx_img, cy_img = w / 2.0, h / 2.0
    diag = float(math.hypot(w, h)) + cfg.math_epsilon
    m8 = (mask.astype(np.uint8) * 255)
    total_orig = int(np.count_nonzero(mask))
    for radius in cfg.bridge_erosion_radii:
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(m8, kernel)
        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if n_labels < 3:  # only background + 1 fg => no split
            continue
        # Gather meaningful blobs
        blobs = [
            {
                "label": lab,
                "area": int(stats[lab, cv2.CC_STAT_AREA]),
                "cx": float(centroids[lab, 0]),
                "cy": float(centroids[lab, 1]),
                "bw": int(stats[lab, cv2.CC_STAT_WIDTH]),
                "bh": int(stats[lab, cv2.CC_STAT_HEIGHT]),
            }
            for lab in range(1, n_labels)
            if int(stats[lab, cv2.CC_STAT_AREA]) >= cfg.bridge_min_split_size
        ]
        if len(blobs) < 2:
            continue
        # Score each blob: product = central+large; support = peripheral+elongated
        for blob in blobs:
            dist = math.hypot(blob["cx"] - cx_img, blob["cy"] - cy_img) / diag
            blob["dist_center"] = dist
            blob["elong"] = float(max(blob["bw"], blob["bh"]) / max(min(blob["bw"], blob["bh"]), 1))
            blob["product_score"] = (blob["area"] / max(total_orig, 1)) * max(0.0, 1.0 - dist * 2.0)
        primary = max(blobs, key=lambda b: b["product_score"])
        # Check if at least one blob is peripheral + elongated (support candidate)
        support_cands = [
            b for b in blobs
            if b["label"] != primary["label"]
            and b["dist_center"] >= cfg.bridge_periphery_threshold
            and b["elong"] >= cfg.bridge_finger_elongation
        ]
        if not support_cands:
            continue
        # Safety: primary blob must be large enough relative to original
        if float(primary["area"] / max(total_orig, 1)) < cfg.bridge_product_min_area_ratio:
            continue
        # Bridge confirmed at this radius
        score = float(np.clip(1.0 - (radius / max(cfg.bridge_erosion_radii[-1], 1)), 0.0, 1.0))
        return score, radius
    return 0.0, 0
def extract_ownership_features(
    region: OwnershipGroupedRegionInput,
    image_hw: tuple[int, int],
    cfg: OwnershipConfig,
) -> OwnershipFeatures:
    """Derive all ownership feature signals for one grouped region."""
    h, w = image_hw
    eps = cfg.math_epsilon
    mask = region.grouped_mask
    diag = float(math.hypot(w, h)) + eps
    image_area = float(max(h * w, 1))
    pixel_area = int(np.count_nonzero(mask))
    relative_area = float(pixel_area / image_area)
    ys, xs = np.where(mask)
    if pixel_area == 0:
        cx, cy = float(w / 2), float(h / 2)
        bbox = (0, 0, 1, 1)
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    center_dist = float(math.hypot(cx - w / 2.0, cy - h / 2.0) / diag)
    border_contact = _border_contact_ratio(mask, eps)
    bw, bh = bbox[2], bbox[3]
    elongation = float(max(bw, bh) / max(min(bw, bh), 1))
    bbox_fill = float(pixel_area / max(bw * bh, eps))
    solidity, complexity = _contour_solidity_complexity(mask)
    irregular = _irregular_boundary_score(mask)
    blob_count, primary_cov, secondary_ratio, finger_ratio = _multi_blob_stats(
        mask, min_area=cfg.bridge_min_split_size // 2
    )
    bridge_score, bridge_radius = _thin_bridge_score(mask, image_hw, cfg)
    return OwnershipFeatures(
        pixel_area=pixel_area,
        relative_area=relative_area,
        centroid_xy=(cx, cy),
        center_distance_norm=center_dist,
        border_contact_ratio=border_contact,
        elongation=elongation,
        solidity=solidity,
        contour_complexity=complexity,
        bbox_fill_ratio=bbox_fill,
        primary_blob_coverage=primary_cov,
        secondary_blob_ratio=secondary_ratio,
        blob_count=blob_count,
        thin_bridge_score=bridge_score,
        bridge_erosion_radius=bridge_radius,
        finger_like_ratio=finger_ratio,
        irregular_boundary_score=irregular,
    )
