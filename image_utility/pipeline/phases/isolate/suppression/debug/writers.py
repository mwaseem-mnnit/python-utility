"""Suppression debug writers (``debug/isolate/suppression/``)."""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from ..config import SuppressionConfig
from ..contracts import SuppressionInput, SuppressionResult

UInt8RGB = NDArray[np.uint8]


def suppression_debug_root() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "suppression")


def write_suppression_debug(
    cfg: SuppressionConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    inp: SuppressionInput,
    result: SuppressionResult,
) -> None:
    if not cfg.debug_enabled:
        return
    root = suppression_debug_root()
    os.makedirs(root, exist_ok=True)

    from .overlays import blend_mask

    removed_removed = frozenset(result.metadata.removed_group_ids)
    removed_union = np.zeros_like(result.combined_survivor_mask, dtype=bool)
    for r in inp.regions:
        if r.group_id in removed_removed:
            removed_union = np.logical_or(removed_union, r.grouped_mask)

    # suppression_removed_regions.png
    rem_canvas = rgb.copy()
    rem_canvas = blend_mask(rem_canvas, removed_union, (220, 60, 60), 0.45)
    cv2.imwrite(
        os.path.join(root, f"{stem}_suppression_removed_regions.png"),
        cv2.cvtColor(rem_canvas, cv2.COLOR_RGB2BGR),
    )

    # suppression_overlay.png
    canvas = rgb.copy()
    top = min(cfg.debug_survivor_cap, len(result.surviving_groups))
    greens = [(40, 200, 80), (30, 180, 120), (60, 210, 100), (50, 190, 90)]
    for i, g in enumerate(result.surviving_groups[:top]):
        canvas = blend_mask(canvas, g.surviving_mask, greens[i % len(greens)], 0.36)
        ys, xs = np.where(g.surviving_mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        lab = f"g{g.group_id} supp={g.suppression_confidence:.2f}"
        cv2.putText(
            canvas,
            lab,
            (x0, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 200),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(root, f"{stem}_suppression_overlay.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    # suppression_survivors.png
    surv = rgb.copy()
    for g in result.surviving_groups[: min(8, len(result.surviving_groups))]:
        surv = blend_mask(surv, g.surviving_mask, (55, 190, 150), 0.42)
    cv2.imwrite(
        os.path.join(root, f"{stem}_suppression_survivors.png"),
        cv2.cvtColor(surv, cv2.COLOR_RGB2BGR),
    )

    payload: dict[str, Any] = {
        "metadata": {
            "analyzed_group_count": result.metadata.analyzed_group_count,
            "removed_group_count": result.metadata.removed_group_count,
            "surviving_group_count": result.metadata.surviving_group_count,
            "removed_group_ids": list(result.metadata.removed_group_ids),
            "global_suppression_confidence": round(result.metadata.global_suppression_confidence, 5),
        },
        "scores": [
            {
                "group_id": s.group_id,
                "artifact_likelihood": round(s.artifact_likelihood, 5),
                "suppression_breakdown": {k: round(float(v), 5) for k, v in s.suppression_breakdown.items()},
            }
            for s in result.scores
        ],
        "survivors": [
            {
                "group_id": g.group_id,
                "member_candidate_ids": list(g.member_candidate_ids),
                "suppression_confidence": round(g.suppression_confidence, 5),
                "removed_region_ids": list(g.removed_region_ids),
                "suppression_breakdown": {k: round(float(v), 5) for k, v in g.suppression_breakdown.items()},
                "geometry_metadata": {
                    k: (round(float(v), 5) if isinstance(v, float) else int(v))
                    for k, v in g.geometry_metadata.items()
                },
            }
            for g in result.surviving_groups
        ],
    }
    with open(os.path.join(root, f"{stem}_suppression_scores.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

