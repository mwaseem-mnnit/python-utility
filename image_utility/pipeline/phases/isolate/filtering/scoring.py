"""Deterministic heuristic scores for semantic proposal filtering."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import FilteringConfig
from .contracts import FilteringProposal, FilteringScore

UInt8RGB = NDArray[np.uint8]
BoolMask = NDArray[np.bool_]


def _border_contact_ratio(mask: BoolMask, eps: float) -> float:
    h, w = mask.shape[:2]
    if h < 2 or w < 2 or not np.any(mask):
        return 0.0
    bd = np.zeros_like(mask, dtype=np.uint8)
    bd[0, :] = 1
    bd[-1, :] = 1
    bd[:, 0] = 1
    bd[:, -1] = 1
    inter = np.logical_and(mask, bd.astype(bool))
    fg = int(np.count_nonzero(mask))
    return float(np.count_nonzero(inter) / max(fg, eps))


def _bbox_from_mask(mask: BoolMask) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, y0, max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)


def _secondary_blob_ratio(mask: BoolMask) -> float:
    m = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return 0.0
    areas = sorted((float(cv2.contourArea(c)) for c in contours), reverse=True)
    if areas[0] < 1:
        return 0.0
    return float(areas[1] / areas[0])


def _convex_solidity(mask: BoolMask) -> float:
    m = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    c = max(contours, key=cv2.contourArea)
    a = float(cv2.contourArea(c))
    if a < 1.0:
        return 1.0
    hull = cv2.convexHull(c)
    ha = float(cv2.contourArea(hull))
    if ha < 1.0:
        return 1.0
    return float(a / ha)


def _detail_focus_on_crop(rgb: UInt8RGB, mask: BoolMask, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return (normalized detail density, normalized focus score) ∈ [0,1] approx."""

    x, y, bw, bh = bbox
    h, w = rgb.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    crop_rgb = rgb[y:y2, x:x2]
    crop_mask = mask[y:y2, x:x2]
    if crop_rgb.size == 0 or not np.any(crop_mask):
        return 0.0, 0.0

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    m = crop_mask.astype(bool)
    lap_v = np.abs(lap[m])
    gm_v = grad_mag[m]
    if lap_v.size == 0:
        return 0.0, 0.0
    # Soft normalize; typical product detail maps to ~0.1–0.4 pre-sigmoid
    detail = float(min(1.0, np.mean(lap_v) / 35.0))
    focus = float(min(1.0, (np.mean(gm_v) / 45.0 + detail) * 0.5))
    return detail, focus


def _stability_signal(p: FilteringProposal) -> float:
    si = float(min(1.0, max(0.0, p.stability_score)))
    iou = float(min(1.0, max(0.0, p.predicted_iou)))
    return float(0.5 * (si + iou))


def score_proposal(
    p: FilteringProposal,
    rgb: UInt8RGB,
    *,
    cfg: FilteringConfig,
) -> FilteringScore:
    """Lightweight deterministic scoring — high rejection_likelihood ⇒ scene-like / invalid."""

    hh, ww = p.mask.shape[:2]
    denom = float(max(hh * ww, cfg.eps))
    area = float(max(p.area, int(np.count_nonzero(p.mask)), 1))
    coverage = float(area / denom)
    border = _border_contact_ratio(p.mask, cfg.eps)
    bbox = _bbox_from_mask(p.mask)
    bbox_fill = float(area / max(bbox[2] * bbox[3], cfg.eps))

    detail, focus = _detail_focus_on_crop(rgb, p.mask, bbox)
    blob2 = _secondary_blob_ratio(p.mask)
    solidity = _convex_solidity(p.mask)
    stab = _stability_signal(p)

    # Coverage slab (giant slabs)
    slab_cov = float(
        max(0.0, min(1.0, (coverage - cfg.max_image_ratio) / max(cfg.eps, 1.0 - cfg.max_image_ratio)))
    )

    # Low detail blur / bokeh
    deficit_detail = float(max(0.0, (cfg.min_detail_density - detail) / max(cfg.min_detail_density, cfg.eps)))
    deficit_focus = float(max(0.0, (cfg.min_focus_score - focus) / max(cfg.min_focus_score, cfg.eps)))

    # Border-only dangerous when fused with slab / blur
    border_weighted = float(border * cfg.border_penalty)

    # Fragment unstable proposal
    frag = float(blob2 / max(cfg.max_blob_ratio, cfg.eps)) if blob2 >= cfg.eps else 0.0
    frag_pen = float(max(0.0, min(1.0, frag)))

    # Amorphous / diffuse blobs: low convex solidity + slab-like coverage cues
    amorph_pen = float(max(0.0, min(1.0, (0.92 - solidity) / 0.5)))
    diffuse_sl = float(min(1.0, slab_cov * max(deficit_detail, deficit_focus) * bbox_fill))

    stability_deficit = float(max(0.0, min(1.0, (0.45 - stab) / 0.45)))

    # Weighted rejection fusion (explainable coefficients baked into terms above)
    terms = np.array(
        [
            slab_cov,
            border_weighted * (0.5 * slab_cov + 0.5 * deficit_detail),
            deficit_detail,
            deficit_focus,
            frag_pen,
            diffuse_sl,
            amorph_pen * deficit_detail,
            stability_deficit * frag_pen,
        ],
        dtype=np.float64,
    )
    w = np.array([1.35, 1.2, 1.05, 0.95, 1.25, 1.35, 0.85], dtype=np.float64)
    fused = float(np.dot(terms[: len(w)], w) / np.sum(w))
    fused = float(max(0.0, min(1.0, fused)))

    rej = fused
    val = float(max(cfg.conf_floor, min(1.0, 1.0 - rej)))

    bd = {
        "coverage_ratio": coverage,
        "border_contact_ratio": border,
        "detail_density_norm": detail,
        "focus_score_norm": focus,
        "bbox_fill_ratio": bbox_fill,
        "secondary_blob_ratio": blob2,
        "contour_convex_solidity": solidity,
        "proposal_stability": stab,
        "slab_cov_penalty": slab_cov,
        "deficit_detail": deficit_detail,
        "deficit_focus": deficit_focus,
        "fragment_penalty": frag_pen,
        "diffuse_scene_penalty": diffuse_sl,
        "stability_frag_penalty": stability_deficit * frag_pen,
    }
    return FilteringScore(
        validity_score=val,
        rejection_likelihood=rej,
        heuristic_breakdown=bd,
    )