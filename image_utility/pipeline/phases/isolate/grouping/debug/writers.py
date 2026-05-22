"""Grouping debug writers (``debug/isolate/grouping/``)."""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from ..config import GroupingConfig
from ..contracts import GroupingResult

UInt8RGB = NDArray[np.uint8]


def grouping_debug_root() -> str:
    return str(WORKSPACE_ROOT / "debug" / "isolate" / "grouping")


def write_grouping_debug(
    cfg: GroupingConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    result: GroupingResult,
    candidate_centroids: dict[int, tuple[float, float]],
) -> None:
    if not cfg.debug_enabled:
        return
    root = grouping_debug_root()
    os.makedirs(root, exist_ok=True)

    from .overlays import blend_mask

    canvas = rgb.copy()
    top_n = max(1, min(cfg.debug_top_groups, len(result.groups)))
    palette = [
        (int(40 + (i * 53) % 200), int(70 + (i * 31) % 200), int(110 + (i * 47) % 200))
        for i in range(max(12, top_n))
    ]
    for i, g in enumerate(result.groups[:top_n]):
        hue = palette[i % len(palette)]
        canvas = blend_mask(canvas, g.grouped_mask, hue, 0.34)
        ys, xs = np.where(g.grouped_mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 220, 50), 1)
        lab = f"g{g.group_id} members={','.join(str(x) for x in g.member_candidate_ids)} conf={g.group_confidence:.2f}"
        cv2.putText(
            canvas,
            lab,
            (x0, max(y0 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 80),
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(root, f"{stem}_grouping_overlay.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    rel_canvas = rgb.copy()

    cand_pos: dict[int, tuple[int, int]] = {
        cid: (int(round(xy[0])), int(round(xy[1]))) for cid, xy in candidate_centroids.items()
    }

    for a, b, aff in result.pair_affinities:
        if aff < cfg.relationship_viz_min_affinity:
            continue
        pa = cand_pos.get(a)
        pb = cand_pos.get(b)
        if pa is None or pb is None:
            continue
        col = (int(80 + 120 * aff), int(200 * (1 - aff)), int(60 + 80 * aff))
        cv2.line(rel_canvas, pa, pb, col, 1, lineType=cv2.LINE_AA)
        mx, my = int((pa[0] + pb[0]) / 2), int((pa[1] + pb[1]) / 2)
        cv2.putText(
            rel_canvas,
            f"{aff:.2f}",
            (mx, my),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            col,
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.imwrite(
        os.path.join(root, f"{stem}_grouping_relationships.png"),
        cv2.cvtColor(rel_canvas, cv2.COLOR_RGB2BGR),
    )

    payload: dict[str, Any] = {
        "metadata": {
            "group_count": result.metadata.group_count,
            "candidate_count": result.metadata.candidate_count,
            "multi_member_group_count": result.metadata.multi_member_group_count,
            "top_group_confidence": round(result.metadata.top_group_confidence, 5),
            "second_group_confidence": round(result.metadata.second_group_confidence, 5),
            "grouping_ambiguity": result.metadata.grouping_ambiguity,
        },
        "pair_affinities": [
            {"a": a, "b": b, "affinity": round(aff, 5)} for a, b, aff in result.pair_affinities
        ],
        "relationships": [
            {
                "candidate_id_a": r.candidate_id_a,
                "candidate_id_b": r.candidate_id_b,
                "overlap_ratio": round(r.overlap_ratio, 5),
                "mask_iou": round(r.mask_iou, 5),
                "centroid_distance_norm": round(r.centroid_distance_norm, 5),
                "bbox_iou": round(r.bbox_iou, 5),
                "containment_smaller_in_larger": round(r.containment_smaller_in_larger, 5),
                "edge_gap_norm": round(r.edge_gap_norm, 5),
                "confidence_similarity": round(r.confidence_similarity, 5),
                "feature_consistency": round(r.feature_consistency, 5),
            }
            for r in result.relationships
        ],
        "groups": [
            {
                "group_id": g.group_id,
                "member_candidate_ids": list(g.member_candidate_ids),
                "group_confidence": round(g.group_confidence, 5),
                "affinity_breakdown": {k: round(float(v), 5) for k, v in g.affinity_breakdown.items()},
                "relationship_pairs": [
                    {"a": p[0], "b": p[1], "affinity": round(float(p[2]), 5)}
                    for p in g.relationship_pairs
                ],
                "geometry_metadata": {k: (round(float(v), 5) if isinstance(v, float) else int(v)) for k, v in g.geometry_metadata.items()},
            }
            for g in result.groups
        ],
    }
    with open(os.path.join(root, f"{stem}_grouping_groups.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    prev = rgb.copy()
    for g in result.groups[: min(6, len(result.groups))]:
        prev = blend_mask(prev, g.grouped_mask, (90, 200, 120), 0.4)
    cv2.imwrite(
        os.path.join(root, f"{stem}_grouping_top_groups.png"),
        cv2.cvtColor(prev, cv2.COLOR_RGB2BGR),
    )

