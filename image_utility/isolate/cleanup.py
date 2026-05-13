"""Morphology and disconnected-fragment removal (artifact cleanup)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

UInt8 = NDArray[np.uint8]


def morphological_pre_cc(mask_255: UInt8, close_ksize: int) -> UInt8:
    """Small closing before CC to merge near-touching product alpha."""
    if close_ksize <= 0:
        return mask_255
    k = close_ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_255, cv2.MORPH_CLOSE, ker)


def remove_tiny_islands(
    binary_mask: UInt8,
    *,
    min_area: int,
) -> UInt8:
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


def strip_small_fragments(
    masked_alpha: UInt8,
    cfg: IsolateConfig,
) -> tuple[UInt8, UInt8, UInt8]:
    """
    Binarize masked alpha, remove sub-threshold islands, return updated alpha + intermediates.

    Intermediates match prior behavior for metrics/debug continuity.
    """
    _retval, bin_frag = cv2.threshold(
        masked_alpha,
        cfg.alpha_visibility_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    clean_bin = remove_tiny_islands(
        bin_frag,
        min_area=cfg.min_fragment_area_after_select,
    )
    new_alpha = np.where(clean_bin > 0, masked_alpha, 0).astype(np.uint8)
    return new_alpha, bin_frag, clean_bin


def morphological_post_open(
    alpha: UInt8,
    open_ksize: int,
    bin_thresh: int,
) -> UInt8:
    """Light opening on binarized alpha to shed thin bridges (off by default; keep conservative)."""
    if open_ksize <= 0:
        return alpha
    k = open_ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    _r, bin_m = cv2.threshold(alpha, bin_thresh, 255, cv2.THRESH_BINARY)
    opened = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, ker)
    return np.where(opened > 0, alpha, 0).astype(np.uint8)
