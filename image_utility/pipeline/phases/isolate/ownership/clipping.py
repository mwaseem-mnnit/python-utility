"""Thin-bridge support clipping — geometry-only, deterministic."""
from __future__ import annotations
import math
import cv2
import numpy as np
from numpy.typing import NDArray
from .config import OwnershipConfig
BoolMask = NDArray[np.bool_]
def clip_support_subregion(
    mask: BoolMask,
    image_hw: tuple[int, int],
    cfg: OwnershipConfig,
) -> tuple[BoolMask, BoolMask, bool]:
    """
    Attempt to separate support sub-regions from the product core via thin-bridge erosion.
    Returns:
        (product_mask, support_mask, clipped)
        - product_mask: the kept (product-side) pixels
        - support_mask: the removed (support-side) pixels
        - clipped: True when an actual split was performed
    """
    h, w = image_hw
    cx_img, cy_img = float(w / 2.0), float(h / 2.0)
    diag = float(math.hypot(w, h)) + cfg.math_epsilon
    m8 = (mask.astype(np.uint8) * 255)
    total_orig = int(np.count_nonzero(mask))
    for radius in cfg.bridge_erosion_radii:
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(m8, kernel)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if n_labels < 3:
            continue
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
        for blob in blobs:
            dist = math.hypot(blob["cx"] - cx_img, blob["cy"] - cy_img) / diag
            blob["dist_center"] = dist
            blob["elong"] = float(max(blob["bw"], blob["bh"]) / max(min(blob["bw"], blob["bh"]), 1))
            # Product score: large + central
            blob["product_score"] = (blob["area"] / max(total_orig, 1)) * max(0.0, 1.0 - dist * 2.0)
        primary = max(blobs, key=lambda b: b["product_score"])
        support_cands = [
            b for b in blobs
            if b["label"] != primary["label"]
            and b["dist_center"] >= cfg.bridge_periphery_threshold
            and b["elong"] >= cfg.bridge_finger_elongation
        ]
        if not support_cands:
            continue
        # Safety guard: primary blob must retain enough of the original mask
        if float(primary["area"] / max(total_orig, 1)) < cfg.bridge_product_min_area_ratio:
            continue
        # Dilate the primary eroded component back to recover pixels
        primary_eroded = (labels == primary["label"]).astype(np.uint8)
        primary_dilated = cv2.dilate(primary_eroded, kernel)
        # Product pixels = original mask AND dilated primary
        product_mask = np.logical_and(mask, primary_dilated.astype(bool))
        support_mask = np.logical_and(mask, ~product_mask)
        return (
            np.ascontiguousarray(product_mask),
            np.ascontiguousarray(support_mask),
            True,
        )
    return (
        np.ascontiguousarray(mask.copy()),
        np.zeros_like(mask),
        False,
    )
