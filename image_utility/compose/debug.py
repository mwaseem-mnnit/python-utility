"""Optional compose debug artifacts."""

from __future__ import annotations

import os

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .config import ComposeConfig

UInt8RGBA = NDArray[np.uint8]


def compose_debug_dir() -> str:
    return str(WORKSPACE_ROOT / "debug" / "compose")


def _ensure_dir() -> None:
    os.makedirs(compose_debug_dir(), exist_ok=True)


def write_compose_debug(
    cfg: ComposeConfig,
    *,
    stem: str,
    source_rgba: UInt8RGBA,
    bbox_xywh: tuple[int, int, int, int],
    cropped_rgba: UInt8RGBA,
    scaled_rgba: UInt8RGBA,
    canvas_hw: tuple[int, int],
    origin_xy: tuple[int, int],
) -> None:
    if not cfg.debug_enabled:
        return
    _ensure_dir()
    root = compose_debug_dir()
    color = cfg.debug_line_color

    # Bbox on alpha preview (BGR for cv2)
    alpha = source_rgba[:, :, 3]
    preview = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    x, y, w, h = bbox_xywh
    cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), color, 2)
    cv2.imwrite(os.path.join(root, f"{stem}_bbox.png"), preview)

    # Checker-free scaled subject preview (flatten onto mid-gray to show halos)
    flat = _flatten_on_gray(scaled_rgba)
    cv2.imwrite(os.path.join(root, f"{stem}_scaled.png"), cv2.cvtColor(flat, cv2.COLOR_RGB2BGR))

    # Placement guide on empty canvas
    ch, cw = canvas_hw
    guide = np.full((ch, cw, 3), (248, 248, 248), dtype=np.uint8)
    ox, oy = origin_xy
    sh, sw = scaled_rgba.shape[:2]
    cv2.rectangle(guide, (ox, oy), (ox + sw - 1, oy + sh - 1), color, 2)
    cx = int(round(cw * cfg.anchor_x_ratio))
    cy = int(round(ch * cfg.anchor_y_ratio))
    cv2.drawMarker(guide, (cx, cy), (180, 180, 180), cv2.MARKER_CROSS, 24, 2)
    cv2.imwrite(os.path.join(root, f"{stem}_placement.png"), guide)


def _flatten_on_gray(rgba: UInt8RGBA, gray: int = 160) -> np.ndarray:
    """RGB preview of RGBA flattened onto solid gray (detect white halos)."""
    a = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    g = np.full_like(rgb, float(gray))
    out = rgb * a[:, :, None] + g * (1.0 - a[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)
