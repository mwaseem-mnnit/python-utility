"""Ranking stage debug writers (``debug/isolate/ranking/``)."""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from ..config import RankingConfig
from ..contracts import RankingResult

UInt8RGB = NDArray[np.uint8]
UInt8 = NDArray[np.uint8]


def ranking_debug_root() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "ranking")


def _features_to_dict(ranked: Any) -> dict[str, Any]:
    f = ranked.features
    return {
        "area": f.area,
        "relative_area": round(f.relative_area, 5),
        "centroid_xy": [round(f.centroid_xy[0], 2), round(f.centroid_xy[1], 2)],
        "bbox_xywh": list(f.bbox_xywh),
        "bbox_fill_ratio": round(f.bbox_fill_ratio, 5),
        "solidity": round(f.solidity, 5),
        "elongation": round(f.elongation, 5),
        "border_contact_ratio": round(f.border_contact_ratio, 5),
        "contour_complexity": round(f.contour_complexity, 5),
        "occupancy_ratio": round(f.occupancy_ratio, 5),
        "center_distance_norm": round(f.center_distance_norm, 5),
        "sam_predicted_iou": round(f.sam_predicted_iou, 5),
        "sam_stability": round(f.sam_stability, 5),
        "overlap_rembg_fg": round(f.overlap_rembg_fg, 5),
    }


def write_ranking_debug(
    cfg: RankingConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    base_alpha: UInt8,
    result: RankingResult,
) -> None:
    if not cfg.debug_enabled:
        return
    root = ranking_debug_root()
    os.makedirs(root, exist_ok=True)

    from .overlays import blend_mask

    canvas = rgb.copy()
    for i, c in enumerate(result.ranked[:12]):
        hue = (
            int(40 + (i * 53) % 200),
            int(70 + (i * 31) % 200),
            int(110 + (i * 47) % 200),
        )
        canvas = blend_mask(canvas, c.mask, hue, 0.32)
        x, y, w, h = c.features.bbox_xywh
        if w > 0 and h > 0:
            cv2.rectangle(
                canvas,
                (x, y),
                (x + w - 1, y + h - 1),
                (255, 255, 0),
                1,
            )
        cv2.putText(
            canvas,
            f"id={c.candidate_id} conf={c.confidence:.2f}",
            (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(root, f"{stem}_ranking_overlay.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    h, w = rgb.shape[:2]
    labels_viz = np.zeros((h, w), dtype=np.uint8)
    for c in result.ranked[: min(254, len(result.ranked))]:
        lab = int((c.candidate_id % 254) + 1)
        labels_viz[c.mask] = lab
    cv2.imwrite(os.path.join(root, f"{stem}_ranking_labels.png"), labels_viz)

    payload = {
        "metadata": {
            "candidate_count": result.metadata.candidate_count,
            "ambiguity_detected": result.metadata.ambiguity_detected,
            "top_confidence": round(result.metadata.top_confidence, 5),
            "second_confidence": round(result.metadata.second_confidence, 5),
            "confidence_separation": round(result.metadata.confidence_separation, 5),
        },
        "candidates": [
            {
                "candidate_id": r.candidate_id,
                "source": r.source,
                "confidence": round(r.confidence, 5),
                "score_breakdown": {k: round(float(v), 5) for k, v in r.score_breakdown.items()},
                "features": _features_to_dict(r),
            }
            for r in result.ranked
        ],
    }
    with open(os.path.join(root, f"{stem}_ranking_scores.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    prev = rgb.copy()
    for c in result.ranked[: min(6, len(result.ranked))]:
        prev = blend_mask(prev, c.mask, (60, 220, 140), 0.38)
    cv2.imwrite(
        os.path.join(root, f"{stem}_ranking_top_candidates.png"),
        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
    )

