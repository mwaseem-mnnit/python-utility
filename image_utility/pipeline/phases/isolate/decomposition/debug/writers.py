"""Decomposition debug artifact writers (stage-local under ``debug/isolate/decomposition/``)."""

from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from ..config import DecompositionConfig
from ..contracts import DecompositionResult, SemanticMaskCandidate
from .overlays import label_colorize, stack_mask_overlay

UInt8RGB = NDArray[np.uint8]


def decomposition_debug_root() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "decomposition")


def write_decomposition_debug(
    cfg: DecompositionConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    result: DecompositionResult,
) -> None:
    if not cfg.debug_enabled:
        return
    root = decomposition_debug_root()
    os.makedirs(root, exist_ok=True)

    cv2.imwrite(os.path.join(root, f"{stem}_base_alpha.png"), result.base_alpha)

    seed = 42
    colored = label_colorize(result.cc_labels, seed=seed)
    cc_viz = np.where(result.cc_labels[..., None] > 0, colored, rgb)
    cv2.imwrite(
        os.path.join(root, f"{stem}_connected_regions.png"),
        cv2.cvtColor(cc_viz, cv2.COLOR_RGB2BGR),
    )

    canvas = rgb.astype(np.float32)
    for i, cand in enumerate(result.semantic_candidates):
        color = (
            int(40 + (i * 37) % 200),
            int(60 + (i * 19) % 200),
            int(80 + (i * 53) % 200),
        )
        canvas = stack_mask_overlay(
            canvas.astype(np.uint8),
            cand.mask,
            color_rgb=color,
            alpha=0.35,
        ).astype(np.float32)

    cv2.imwrite(
        os.path.join(root, f"{stem}_semantic_masks.png"),
        cv2.cvtColor(canvas.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    if result.semantic_candidates:
        union = np.zeros(rgb.shape[:2], dtype=bool)
        for c in result.semantic_candidates:
            union |= c.mask
        u = stack_mask_overlay(rgb, union, color_rgb=(120, 220, 100), alpha=0.4)
        cv2.imwrite(
            os.path.join(root, f"{stem}_candidate_union.png"),
            cv2.cvtColor(u, cv2.COLOR_RGB2BGR),
        )

    prev = rgb.copy()
    for i, alpha_c in enumerate(result.alpha_candidates[: min(8, len(result.alpha_candidates))]):
        m = alpha_c > cfg.alpha_visibility_threshold
        prev = stack_mask_overlay(prev, m, color_rgb=(255, 100, 180), alpha=0.25)
    cv2.imwrite(
        os.path.join(root, f"{stem}_candidate_previews.png"),
        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
    )


def candidate_row_preview(
    rgb: UInt8RGB,
    candidates: Iterable[SemanticMaskCandidate],
    *,
    max_cols: int = 4,
    thumb: int = 128,
) -> UInt8RGB:
    """Optional small grid (not used in default writers; available for tooling)."""
    rows: list[list[np.ndarray]] = []
    row: list[np.ndarray] = []
    for i, c in enumerate(candidates):
        ov = stack_mask_overlay(rgb, c.mask, (0, 255, 128), 0.45)
        ov = cv2.resize(ov, (thumb, thumb), interpolation=cv2.INTER_AREA)
        row.append(ov)
        if len(row) >= max_cols:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    if not rows:
        return np.zeros((thumb, thumb, 3), dtype=np.uint8)
    return np.vstack([np.hstack(r) for r in rows])
