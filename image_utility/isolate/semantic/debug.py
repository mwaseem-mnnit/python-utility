"""Semantic (SAM) debug PNGs under ``debug/isolate/semantic/``."""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from ..config import IsolateConfig
from .ranking import SemanticRegionScore

UInt8RGB = NDArray[np.uint8]


def semantic_debug_dir() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "semantic")


def _ensure(dir_path: str) -> None:
    os.makedirs(dir_path, exist_ok=True)


def _to_bgr(rgb: UInt8RGB) -> NDArray[np.uint8]:
    if rgb.ndim == 2:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    if rgb.shape[2] == 4:
        return cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_semantic_debug(
    cfg: IsolateConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    scored: Sequence[SemanticRegionScore],
    selected: SemanticRegionScore | None,
) -> None:
    if not cfg.debug_enabled:
        return
    root = semantic_debug_dir()
    _ensure(root)
    base = _to_bgr(rgb)
    ih, iw = base.shape[:2]

    # All regions pseudo-color overlay
    canvas = base.copy()
    rng = np.random.default_rng(cfg.debug_color_seed + 17)
    if scored:
        blend = float(np.clip(cfg.semantic_debug_mask_blend, 0.05, 0.7))
        for s in scored:
            color = tuple(int(x) for x in rng.integers(48, 255, size=3, dtype=np.uint8))
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            m = s.mask
            for c in range(3):
                canvas[:, :, c] = np.where(
                    m,
                    (canvas[:, :, c].astype(np.float32) * (1 - blend) + color_bgr[c] * blend).astype(np.uint8),
                    canvas[:, :, c],
                )
    cv2.imwrite(os.path.join(root, f"{stem}_sam_masks.png"), canvas)

    # Bboxes + confidence
    ann = base.copy()
    fs = float(max(0.35, min(ih, iw) / 900.0))
    th = max(1, int(round(fs * 2)))
    for s in scored:
        ys, xs = np.where(s.mask)
        if ys.size < 1:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        is_sel = selected is not None and s.region_id == selected.region_id
        col = (0, 255, 0) if is_sel else (0, 0, 255)
        thick = 3 if is_sel else 1
        cv2.rectangle(ann, (x0, y0), (x1, y1), col, thick)
        tag = f"id={s.region_id} c={s.confidence:.2f}"
        cv2.putText(
            ann,
            tag,
            (x0, max(y0 - 4, int(16 * fs))),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            col,
            th,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(os.path.join(root, f"{stem}_semantic_rank.png"), ann)
