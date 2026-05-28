"""Ownership debug writers — outputs to debug/isolate/ownership/."""
from __future__ import annotations
import json
import os
from typing import Any
import cv2
import numpy as np
from numpy.typing import NDArray
from image_utility.config import WORKSPACE_ROOT
from ..config import OwnershipConfig
from ..contracts import OwnershipInput, OwnershipResult
UInt8RGB = NDArray[np.uint8]
def ownership_debug_root() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "ownership")
def write_ownership_debug(
    cfg: OwnershipConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    inp: OwnershipInput,
    result: OwnershipResult,
) -> None:
    if not cfg.debug_enabled:
        return
    root = ownership_debug_root()
    os.makedirs(root, exist_ok=True)
    from .overlays import LABEL_COLOURS, blend_mask
    top = min(cfg.debug_top_groups, len(result.owned_groups))
    # 1. ownership_overlay.png — colour-coded by label
    canvas = rgb.copy()
    for g in result.owned_groups[:top]:
        colour = LABEL_COLOURS.get(g.ownership_label, (180, 180, 180))
        canvas = blend_mask(canvas, g.surviving_mask, colour, 0.38)
        ys, xs = np.where(g.surviving_mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        label_text = f"g{g.group_id} {g.ownership_label[:4]} {g.ownership_confidence:.2f}"
        cv2.putText(
            canvas,
            label_text,
            (x0, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 200),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(root, f"{stem}_ownership_overlay.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )
    # 2. ownership_support_removed.png — red overlay on removed pixels
    removed_union = np.zeros(rgb.shape[:2], dtype=bool)
    for g in result.owned_groups:
        removed_union = np.logical_or(removed_union, g.removed_support_mask)
    if np.any(removed_union):
        rem_canvas = blend_mask(rgb.copy(), removed_union, (220, 60, 60), 0.5)
        cv2.imwrite(
            os.path.join(root, f"{stem}_ownership_support_removed.png"),
            cv2.cvtColor(rem_canvas, cv2.COLOR_RGB2BGR),
        )
    # 3. ownership_product_mask.png — combined product mask
    prod_canvas = blend_mask(rgb.copy(), result.combined_product_mask, (40, 200, 80), 0.45)
    cv2.imwrite(
        os.path.join(root, f"{stem}_ownership_product_mask.png"),
        cv2.cvtColor(prod_canvas, cv2.COLOR_RGB2BGR),
    )
    # 4. ownership_scores.json
    payload: dict[str, Any] = {
        "metadata": {
            "analyzed_group_count": result.metadata.analyzed_group_count,
            "product_group_count": result.metadata.product_group_count,
            "support_group_count": result.metadata.support_group_count,
            "packaging_group_count": result.metadata.packaging_group_count,
            "environment_group_count": result.metadata.environment_group_count,
            "uncertain_group_count": result.metadata.uncertain_group_count,
            "global_ownership_confidence": round(result.metadata.global_ownership_confidence, 5),
            "primary_product_group_id": result.metadata.primary_product_group_id,
            "support_clipped_group_count": result.metadata.support_clipped_group_count,
        },
        "scores": [
            {
                "group_id": s.group_id,
                "assigned_label": s.assigned_label,
                "product_likelihood": round(s.product_likelihood, 5),
                "support_likelihood": round(s.support_likelihood, 5),
                "packaging_likelihood": round(s.packaging_likelihood, 5),
                "environment_likelihood": round(s.environment_likelihood, 5),
                "score_breakdown": {k: round(float(v), 5) for k, v in s.score_breakdown.items()},
                "features": {
                    "pixel_area": s.features.pixel_area,
                    "relative_area": round(s.features.relative_area, 5),
                    "center_distance_norm": round(s.features.center_distance_norm, 5),
                    "border_contact_ratio": round(s.features.border_contact_ratio, 5),
                    "elongation": round(s.features.elongation, 5),
                    "solidity": round(s.features.solidity, 5),
                    "contour_complexity": round(s.features.contour_complexity, 5),
                    "bbox_fill_ratio": round(s.features.bbox_fill_ratio, 5),
                    "blob_count": s.features.blob_count,
                    "primary_blob_coverage": round(s.features.primary_blob_coverage, 5),
                    "secondary_blob_ratio": round(s.features.secondary_blob_ratio, 5),
                    "thin_bridge_score": round(s.features.thin_bridge_score, 5),
                    "bridge_erosion_radius": s.features.bridge_erosion_radius,
                    "finger_like_ratio": round(s.features.finger_like_ratio, 5),
                    "irregular_boundary_score": round(s.features.irregular_boundary_score, 5),
                },
            }
            for s in result.scores
        ],
        "groups": [
            {
                "group_id": g.group_id,
                "ownership_label": g.ownership_label,
                "ownership_confidence": round(g.ownership_confidence, 5),
                "product_likelihood": round(g.product_likelihood, 5),
                "support_likelihood": round(g.support_likelihood, 5),
                "member_candidate_ids": list(g.member_candidate_ids),
                "surviving_px": int(np.count_nonzero(g.surviving_mask)),
                "removed_support_px": int(np.count_nonzero(g.removed_support_mask)),
            }
            for g in result.owned_groups
        ],
    }
    with open(os.path.join(root, f"{stem}_ownership_scores.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
