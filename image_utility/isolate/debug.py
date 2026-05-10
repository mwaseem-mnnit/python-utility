"""Optional debug artifacts for isolate tuning."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from image_utility.config import WORKSPACE_ROOT


def isolate_debug_dir() -> Path:
    return WORKSPACE_ROOT / "debug" / "isolate"


def save_alpha_png(stem: str, alpha: np.ndarray) -> None:
    path = isolate_debug_dir() / f"{stem}_alpha.png"
    isolate_debug_dir().mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), alpha)


def save_components_viz(stem: str, labels: np.ndarray) -> None:
    """Pseudo-color connected-component labels."""
    isolate_debug_dir().mkdir(parents=True, exist_ok=True)
    n = int(labels.max()) + 1
    if n <= 1:
        viz = np.zeros((*labels.shape, 3), dtype=np.uint8)
    else:
        lut = np.random.default_rng(42).integers(0, 255, size=(n, 3), dtype=np.uint8)
        lut[0] = (0, 0, 0)
        viz = lut[labels]
    path = isolate_debug_dir() / f"{stem}_components.png"
    cv2.imwrite(str(path), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))


def save_selection_overlay(stem: str, rgb: np.ndarray, labels: np.ndarray, keep: int) -> None:
    """Draw semi-transparent green over selected component."""
    isolate_debug_dir().mkdir(parents=True, exist_ok=True)
    base = rgb.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.shape[2] == 4:
        base = cv2.cvtColor(base, cv2.COLOR_RGBA2BGR)
    else:
        base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    sel = (labels == keep).astype(np.float32)
    green = np.zeros_like(base, dtype=np.float32)
    green[:, :, 1] = 255.0
    overlay = base.astype(np.float32)
    blend = 0.35
    out = overlay * (1 - sel[:, :, None] * blend) + green * (sel[:, :, None] * blend)
    path = isolate_debug_dir() / f"{stem}_selected.png"
    cv2.imwrite(str(path), np.clip(out, 0, 255).astype(np.uint8))
