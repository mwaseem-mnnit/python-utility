"""Optional debug artifacts (config-gated)."""

from __future__ import annotations

import os

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .config import IsolateConfig
from .components import Labels

UInt8 = NDArray[np.uint8]
UInt8RGB = NDArray[np.uint8]


def isolate_debug_dir() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate")


def _ensure_debug_dir() -> None:
    os.makedirs(isolate_debug_dir(), exist_ok=True)


def write_isolate_debug(
    cfg: IsolateConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    labels: Labels,
    keep_label: int,
    refined_alpha: UInt8,
) -> None:
    """Write debug PNGs when ``cfg.debug_enabled``."""
    if not cfg.debug_enabled:
        return
    _ensure_debug_dir()
    root = isolate_debug_dir()
    cv2.imwrite(os.path.join(root, f"{stem}_alpha.png"), refined_alpha)
    _save_components_viz(cfg, os.path.join(root, f"{stem}_components.png"), labels)
    _save_selection_overlay(
        cfg,
        os.path.join(root, f"{stem}_selected.png"),
        rgb,
        labels,
        keep_label,
    )


def _save_components_viz(cfg: IsolateConfig, path: str, labels: Labels) -> None:
    n = int(labels.max()) + 1
    if n <= 1:
        viz = np.zeros((*labels.shape, 3), dtype=np.uint8)
    else:
        rng = np.random.default_rng(cfg.debug_color_seed)
        lut = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
        lut[0] = (0, 0, 0)
        viz = lut[labels]
    cv2.imwrite(path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


def _save_selection_overlay(
    cfg: IsolateConfig,
    path: str,
    rgb: UInt8RGB,
    labels: Labels,
    keep_label: int,
) -> None:
    base = rgb.copy()
    if base.ndim == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.shape[2] == 4:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGBA2BGR)
    else:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    sel = (labels == keep_label).astype(np.float32)
    green = np.zeros_like(base_bgr, dtype=np.float32)
    green[:, :, 1] = 255.0
    overlay = base_bgr.astype(np.float32)
    blend = cfg.debug_overlay_blend
    out = overlay * (1 - sel[:, :, None] * blend) + green * (sel[:, :, None] * blend)
    cv2.imwrite(path, np.clip(out, 0, 255).astype(np.uint8))
