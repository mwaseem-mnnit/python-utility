"""Optional debug artifacts (config-gated)."""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .components import (
    SEMANTIC_KEEP,
    SEMANTIC_REJECT,
    ComponentFeatures,
    Labels,
)
from .config import IsolateConfig

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
    ranked: Sequence[ComponentFeatures] | None = None,
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
    if ranked:
        _save_rank_overlay(
            cfg,
            os.path.join(root, f"{stem}_rank_overlay.png"),
            rgb,
            labels,
            keep_label,
            ranked,
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


def _to_bgr(rgb: UInt8RGB) -> UInt8:
    base = rgb.copy()
    if base.ndim == 2:
        return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if base.shape[2] == 4:
        return cv2.cvtColor(base, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(base, cv2.COLOR_RGB2BGR)


def _save_rank_overlay(
    cfg: IsolateConfig,
    path: str,
    rgb: UInt8RGB,
    labels: Labels,
    keep_label: int,
    ranked: Sequence[ComponentFeatures],
) -> None:
    canvas = _to_bgr(rgb)
    ih, iw = canvas.shape[:2]
    fs = float(max(0.35, min(ih, iw) / 900.0))
    thick_txt = max(1, int(round(fs * 2)))

    cv2.rectangle(canvas, (0, 0), (iw - 1, ih - 1), (180, 180, 180), 2)

    for f in ranked:
        x, y, w, h = f.bbox
        x2, y2 = min(x + w, iw - 1), min(y + h, ih - 1)
        if f.semantic == SEMANTIC_KEEP:
            col = (0, 255, 0)
            th = 3
        elif f.semantic == SEMANTIC_REJECT:
            col = (0, 0, 255)
            th = 1
        else:
            col = (0, 255, 255)
            th = 2
        cv2.rectangle(canvas, (x, y), (x2, y2), col, th)
        txt = f"L{f.label} c={f.confidence:.2f} b={f.border_contact_ratio:.2f} {f.semantic}"
        ty = max(y - 4, int(18 * fs) + 2)
        cv2.putText(
            canvas,
            txt,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            col,
            thick_txt,
            lineType=cv2.LINE_AA,
        )

    winner = next((f for f in ranked if f.label == keep_label), None)
    if winner is not None and winner.border_contact_ratio > 1e-6:
        m = labels == keep_label
        edge = np.zeros_like(m, dtype=bool)
        edge[0, :] |= m[0, :]
        edge[-1, :] |= m[-1, :]
        edge[:, 0] |= m[:, 0]
        edge[:, -1] |= m[:, -1]
        canvas[edge] = (255, 0, 255)

    cv2.imwrite(path, canvas)
