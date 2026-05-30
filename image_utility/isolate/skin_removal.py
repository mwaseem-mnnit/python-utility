"""Skin/hand detection and removal from product foreground.

Uses spatial analysis (vertical skin-density profiling) to find the hand
boundary, then removes everything below it and inpaints gaps where the hand
was gripping the product.

Key insight: metallic/chrome products heavily overlap with skin color ranges,
so skin color CANNOT be used to decide what to keep vs remove in the product
zone.  Only spatial position (above vs below boundary) determines the cut.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

LOGGER = logging.getLogger(__name__)

UInt8 = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Color-space skin detection (used ONLY for boundary-finding, not removal)
# ---------------------------------------------------------------------------

def _broad_skin_mask(img_bgr: UInt8) -> BoolMask:
    """Union of YCrCb and HSV skin detectors (broad, high recall)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    skin_cr = (
        (ycrcb[:, :, 1] >= 130) & (ycrcb[:, :, 1] <= 175) &
        (ycrcb[:, :, 2] >= 75) & (ycrcb[:, :, 2] <= 130)
    )
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    skin_hs = (
        (hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) &
        (hsv[:, :, 1] >= 35) & (hsv[:, :, 1] <= 180) &
        (hsv[:, :, 2] >= 55)
    )
    return skin_cr | skin_hs


def _resolution_scale(h: int, w: int) -> int:
    return max(1, int(min(h, w) / 500))


# ---------------------------------------------------------------------------
# Vertical density analysis → hand boundary
# ---------------------------------------------------------------------------

def _find_hand_boundary(
    fg_mask: BoolMask,
    skin_in_fg: BoolMask,
    h: int,
) -> int | None:
    """
    Find the y-coordinate where the hand begins using row-wise skin density.

    Returns None if no clear hand boundary is detected.

    Metallic products have skin-density ~0.50-0.67 (warm reflections).
    Actual hand/skin regions have density ~0.75-1.0.
    We look for the first sustained run of rows with density > 0.75.
    """
    fg_ys = np.where(fg_mask)[0]
    if len(fg_ys) == 0:
        return None

    y_min_fg = int(fg_ys.min())
    y_max_fg = int(fg_ys.max())
    n_rows = y_max_fg - y_min_fg + 1

    if n_rows < 100:
        return None

    # Compute per-row skin density
    density = np.zeros(n_rows, dtype=np.float32)
    for iy in range(n_rows):
        row = y_min_fg + iy
        fg_r = int(np.count_nonzero(fg_mask[row, :]))
        sk_r = int(np.count_nonzero(skin_in_fg[row, :]))
        if fg_r > 0:
            density[iy] = sk_r / fg_r

    # Smooth to avoid noise
    density_smooth = cv2.GaussianBlur(
        density.reshape(-1, 1), (1, 51), 0
    ).flatten()

    # Find sustained high density (hand zone): 50+ consecutive rows > 0.75
    threshold = 0.75
    run_len = 50
    above = density_smooth > threshold
    min_search = int(n_rows * 0.30)  # Skip top 30% (definitely product)

    for i in range(min_search, len(above) - run_len):
        if all(above[i:i + run_len]):
            # Transition found — use it directly as boundary (small margin above)
            boundary_y = y_min_fg + i - 10
            boundary_y = max(y_min_fg, boundary_y)
            LOGGER.info(
                "[skin] density transition at row %d/%d (y=%d), density=%.2f",
                i, n_rows, boundary_y, density_smooth[i],
            )
            return boundary_y

    # No clear transition → no hand detected
    return None


# ---------------------------------------------------------------------------
# Spatial hand removal with inpainting
# ---------------------------------------------------------------------------

def detect_and_remove_skin(
    rgb: UInt8,
    alpha: UInt8,
    cfg: IsolateConfig,
) -> tuple[UInt8, dict]:
    """
    Detect and remove hand/finger regions from the product alpha mask.

    Strategy (spatial, NOT color-based removal):
    1. Use skin color density per row to find the hand boundary
    2. ABOVE boundary: keep ALL foreground (product zone — never remove by color)
    3. BELOW boundary: remove ALL foreground (hand zone)
    4. Transition zone: taper product projection to recover product tail
    5. Inpaint gaps where hand was gripping the product
    6. Extra erosion at bottom edge to eliminate thin hand fringe
    """
    h, w = alpha.shape[:2]
    meta: dict = {"skin_removal_applied": False, "skin_px_removed": 0}

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fg_mask = alpha > cfg.alpha_visibility_threshold
    fg_px = int(np.count_nonzero(fg_mask))

    if fg_px == 0:
        return alpha, meta

    # Broad skin detection (for boundary analysis only)
    skin_broad = _broad_skin_mask(bgr) & fg_mask
    skin_ratio = float(np.count_nonzero(skin_broad)) / fg_px
    meta["skin_ratio_in_fg"] = round(skin_ratio, 4)

    if skin_ratio < cfg.skin_min_ratio:
        LOGGER.info("[skin] minimal skin (%.1f%%), skipping", skin_ratio * 100)
        return alpha, meta

    if skin_ratio > cfg.skin_max_ratio:
        LOGGER.warning("[skin] excessive skin (%.1f%%), skipping", skin_ratio * 100)
        meta["skin_skip_reason"] = "excessive_ratio"
        return alpha, meta

    # Find hand boundary via vertical density
    boundary_y = _find_hand_boundary(fg_mask, skin_broad, h)

    if boundary_y is None:
        LOGGER.info("[skin] no hand boundary detected, skipping")
        meta["skin_skip_reason"] = "no_boundary"
        return alpha, meta

    meta["hand_boundary_y"] = boundary_y
    scale = _resolution_scale(h, w)

    # ── Build product mask: keep above boundary, remove below ────────
    product_mask = fg_mask.copy()
    product_mask[boundary_y:, :] = False

    # Transition zone: recover product tail below boundary using projection
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    rows_above = 20
    product_x_ranges = []
    for dy in range(rows_above):
        row = boundary_y - 1 - dy
        if row < 0:
            continue
        fg_cols = np.where(fg_mask[row, :])[0]
        if len(fg_cols) > 0:
            product_x_ranges.append((int(fg_cols.min()), int(fg_cols.max())))

    y_max_fg = int(np.where(fg_mask)[0].max())
    if product_x_ranges:
        avg_x_min = int(np.mean([r[0] for r in product_x_ranges]))
        avg_x_max = int(np.mean([r[1] for r in product_x_ranges]))
        pw = avg_x_max - avg_x_min
        transition_depth = min(80, y_max_fg - boundary_y)
        for dy in range(transition_depth):
            row = boundary_y + dy
            if row >= h:
                break
            shrink = dy / max(transition_depth, 1)
            xc = (avg_x_min + avg_x_max) / 2
            half = pw * (1 - shrink * 0.7) / 2
            x_lo, x_hi = int(xc - half), int(xc + half)
            for x in range(max(0, x_lo), min(w, x_hi)):
                if fg_mask[row, x]:
                    cr_v = int(ycrcb[row, x, 1])
                    cb_v = int(ycrcb[row, x, 2])
                    hue_v = int(hsv_img[row, x, 0])
                    sat_v = int(hsv_img[row, x, 1])
                    is_skin = (135 <= cr_v <= 165 and 82 <= cb_v <= 120
                               and hue_v <= 20 and sat_v >= 50)
                    if not is_skin:
                        product_mask[row, x] = True

    # ── Morphological cleanup ────────────────────────────────────────
    pm_u8 = product_mask.astype(np.uint8) * 255
    ck = 15 * scale
    pm_u8 = cv2.morphologyEx(pm_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck | 1, ck | 1)))

    # Fill internal holes
    contours, hier = cv2.findContours(pm_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for i in range(len(contours)):
            if hier[0][i][3] != -1:
                cv2.drawContours(pm_u8, contours, i, 255, thickness=cv2.FILLED)

    # Keep largest component
    nl, lb, st, _ = cv2.connectedComponentsWithStats(pm_u8, connectivity=8)
    if nl > 2:
        areas = st[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        pm_u8 = ((lb == best).astype(np.uint8) * 255)

    # Extra erosion at bottom edge to kill thin hand fringe
    pys_arr = np.where(pm_u8 > 0)[0]
    if len(pys_arr) > 0:
        prod_top = int(pys_arr.min())
        prod_bot = int(pys_arr.max())
        erode_start = prod_bot - int((prod_bot - prod_top) * 0.08)
        # Standard light erosion everywhere
        ek = 2 * scale
        pm_u8 = cv2.erode(pm_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek | 1, ek | 1)))
        # Extra aggressive erosion at bottom
        bottom_part = np.zeros_like(pm_u8)
        bottom_part[erode_start:, :] = pm_u8[erode_start:, :]
        extra_ek = 8 * scale
        bottom_eroded = cv2.erode(bottom_part,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (extra_ek | 1, extra_ek | 1)))
        pm_u8[erode_start:, :] = bottom_eroded[erode_start:, :]
        # Dilate back slightly
        dk = 1 * scale
        pm_u8 = cv2.dilate(pm_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk | 1, dk | 1)))

    # Stay within original fg
    pm_u8 = pm_u8 & (fg_mask.astype(np.uint8) * 255)

    # ── Inpainting: fill where hand gripped the product ──────────────
    pm_bool = pm_u8 > 0
    contours_p, _ = cv2.findContours(pm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    if contours_p:
        hull = cv2.convexHull(max(contours_p, key=cv2.contourArea))
        cv2.drawContours(hull_mask, [hull], 0, 255, thickness=cv2.FILLED)

    inpaint_region = (hull_mask > 0) & ~pm_bool & fg_mask
    border_k = 5 * scale
    prod_border = cv2.dilate(pm_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_k | 1, border_k | 1)))
    inpaint_region = inpaint_region | ((prod_border > 0) & ~pm_bool & fg_mask)
    inpaint_u8 = inpaint_region.astype(np.uint8) * 255
    inpaint_px = int(np.count_nonzero(inpaint_region))

    if inpaint_px > 0:
        inpainted_bgr = cv2.inpaint(bgr, inpaint_u8, inpaintRadius=12, flags=cv2.INPAINT_NS)
        inpainted_bgr = cv2.inpaint(inpainted_bgr, inpaint_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        # Write inpainted pixels back into the source
        mask_3ch = np.stack([inpaint_region] * 3, axis=-1)
        rgb[mask_3ch] = inpainted_rgb[mask_3ch]
        LOGGER.info("[skin] inpainted %d px in product grip zone", inpaint_px)

    # Update alpha to include inpainted area
    alpha_out = pm_u8.copy()
    alpha_out[inpaint_region] = 255
    alpha_out = cv2.GaussianBlur(alpha_out.astype(np.float32), (0, 0), sigmaX=1.0)
    alpha_out = np.clip(alpha_out, 0, 255).astype(np.uint8)

    # Safety: don't destroy the product
    remaining = int(np.count_nonzero(alpha_out > cfg.alpha_visibility_threshold))
    if remaining < fg_px * 0.15:
        LOGGER.warning("[skin] would destroy product (%.1f%% remaining), aborting",
                       100 * remaining / fg_px)
        meta["skin_skip_reason"] = "would_destroy_product"
        return alpha, meta

    removed_px = fg_px - remaining
    meta["skin_removal_applied"] = True
    meta["skin_px_removed"] = removed_px
    meta["skin_pct_removed"] = round(100 * removed_px / fg_px, 1)
    meta["inpainted_px"] = inpaint_px
    meta["strategy"] = "spatial_boundary"

    LOGGER.info(
        "[skin] removed %d px (%.1f%%), boundary y=%d, inpainted %d px",
        removed_px, meta["skin_pct_removed"], boundary_y, inpaint_px,
    )

    return alpha_out, meta
