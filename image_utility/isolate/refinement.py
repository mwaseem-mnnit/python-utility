"""Alpha edge refinement, defringe, and final RGBA assembly."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import IsolateConfig

UInt8 = NDArray[np.uint8]
UInt8RGBA = NDArray[np.uint8]


def refine_alpha_soft(alpha: UInt8, sigma: float) -> UInt8:
    """Mild Gaussian smoothing on alpha (feather edges)."""
    if sigma <= 0:
        return alpha
    a = alpha.astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


def _decontaminate_fringe(
    rgb: UInt8,
    alpha: UInt8,
    *,
    low_alpha: int = 180,
    radius: int = 7,
    dark_threshold: int = 30,
) -> UInt8:
    """
    Fix two kinds of dark contamination in the product RGBA:

    1. **Semi-transparent edge pixels** (alpha in 0..low_alpha): replace RGB
       with average of nearby opaque product pixels (standard defringe).

    2. **Opaque dark background bleed** (alpha >= low_alpha but RGB near-black):
       rembg sometimes classifies dark store backgrounds as foreground.
       These fully-opaque but very dark pixels get their RGB replaced with
       nearby bright product color via weighted blur.

    Both are fixed by propagating the interior product color outward.
    """
    h, w = alpha.shape[:2]

    # Pixels that are semi-transparent edges
    edge_mask = (alpha > 0) & (alpha < low_alpha)

    # Pixels that are fully opaque but suspiciously dark (background bleed)
    # A real product pixel is rarely pure black; these are rembg artifacts
    opaque_mask = alpha >= low_alpha
    rgb_brightness = rgb.max(axis=2)  # max channel value
    dark_opaque = opaque_mask & (rgb_brightness < dark_threshold)

    # Combined mask of pixels that need color replacement
    fix_mask = edge_mask | dark_opaque
    if not np.any(fix_mask):
        return rgb

    # Source: fully opaque AND sufficiently bright (real product interior)
    solid_bright = opaque_mask & (rgb_brightness >= dark_threshold)
    if not np.any(solid_bright):
        return rgb

    # Weighted blur: only solid bright pixels contribute color
    rgb_f = rgb.astype(np.float32)
    solid_f = solid_bright.astype(np.float32)
    weighted_rgb = rgb_f * solid_f[:, :, np.newaxis]

    k = radius * 2 + 1
    blurred_rgb = cv2.blur(weighted_rgb, (k, k))
    blurred_weight = cv2.blur(solid_f, (k, k))

    safe_weight = np.maximum(blurred_weight, 1e-6)
    interior_color = np.clip(blurred_rgb / safe_weight[:, :, np.newaxis], 0, 255)

    # For dark opaque pixels where the local blur has no nearby bright pixels,
    # do a second pass with a much larger radius to reach further
    still_dark = fix_mask & (blurred_weight < 0.01)
    if np.any(still_dark):
        big_k = radius * 8 + 1
        blurred_rgb_big = cv2.blur(weighted_rgb, (big_k, big_k))
        blurred_weight_big = cv2.blur(solid_f, (big_k, big_k))
        safe_big = np.maximum(blurred_weight_big, 1e-6)
        far_color = np.clip(blurred_rgb_big / safe_big[:, :, np.newaxis], 0, 255)
        interior_color[still_dark] = far_color[still_dark]

    out = rgb.copy()
    out[fix_mask] = interior_color[fix_mask].astype(np.uint8)
    return out


def compose_isolated_rgba(
    original_rgba: UInt8RGBA,
    new_alpha: UInt8,
    cfg: IsolateConfig,
) -> UInt8RGBA:
    """
    Build RGBA with cleaned alpha; decontaminate edge RGB to prevent dark fringe.

    Zero RGB where alpha is effectively zero. For semi-transparent edge pixels,
    push RGB toward the interior product color to eliminate dark patches.
    """
    out = np.zeros_like(original_rgba)
    out[:, :, 3] = new_alpha

    rgb = original_rgba[:, :, :3].copy()

    # Decontaminate fringe: replace edge pixel colors with interior product color
    rgb = _decontaminate_fringe(rgb, new_alpha)

    visible = new_alpha[:, :, np.newaxis] > cfg.rgb_zero_below_alpha
    out[:, :, :3] = np.where(visible, rgb, 0).astype(np.uint8)
    return out
