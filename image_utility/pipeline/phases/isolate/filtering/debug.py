"""Filtering debug artifacts under ``debug/isolate/filtering/``."""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .config import FilteringConfig
from .contracts import FilteringInput, FilteringResult

UInt8RGB = NDArray[np.uint8]


def _blend(
    rgb: UInt8RGB,
    mask: NDArray[np.bool_],
    color: tuple[int, int, int],
    strength: float,
) -> UInt8RGB:
    out = rgb.astype(np.float32)
    c = np.array(color, dtype=np.float32)
    m = mask.astype(np.float32)[:, :, None]
    s = float(max(0.0, min(1.0, strength)))
    out = out * (1 - m * s) + c * (m * s)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_filtering_debug(
    cfg: FilteringConfig,
    *,
    stem: str,
    rgb: UInt8RGB,
    inp: FilteringInput,
    result: FilteringResult,
) -> None:
    if not cfg.debug_enabled:
        return

    root = os.path.join(str(WORKSPACE_ROOT), "debug", "isolate", "filtering")
    os.makedirs(root, exist_ok=True)

    hh, ww = inp.image_hw
    accept_union = np.zeros((hh, ww), dtype=bool)
    reject_union = np.zeros((hh, ww), dtype=bool)

    acc_ids = {p.candidate_id for p in result.accepted}
    rej_ids = {p.candidate_id for p in result.rejected}

    for p in inp.proposals:
        if p.candidate_id in acc_ids:
            accept_union = np.logical_or(accept_union, p.mask)
        if p.candidate_id in rej_ids:
            reject_union = np.logical_or(reject_union, p.mask)

    survivor_img = rgb.copy()
    survivor_img = _blend(survivor_img, accept_union, (48, 200, 90), 0.38)

    rej_img = rgb.copy()
    rej_img = _blend(rej_img, reject_union, (220, 72, 58), 0.42)

    overlay = rgb.copy()
    overlay = _blend(overlay, accept_union, (40, 210, 100), 0.32)
    overlay = _blend(overlay, reject_union, (230, 50, 65), 0.30)

    for i, sp in enumerate(result.scored[: cfg.overlay_top_n]):
        p = sp.proposal
        ys, xs = np.where(p.mask)
        if len(xs) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        is_acc = p.candidate_id in acc_ids
        col = (40, 255, 140) if is_acc else (255, 160, 60)
        label = (
            f"id={p.candidate_id} v={sp.filtering_score.validity_score:.2f} "
            f"rej={sp.filtering_score.rejection_likelihood:.2f}"
        )
        cv2.putText(
            overlay,
            label,
            (x0 + (i % 3) * 4, max(y0 - 4 - (i // 3) * 14, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            col,
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(os.path.join(root, f"{stem}_filtering_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        os.path.join(root, f"{stem}_filtering_survivors.png"),
        cv2.cvtColor(survivor_img, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(root, f"{stem}_filtering_rejected.png"),
        cv2.cvtColor(rej_img, cv2.COLOR_RGB2BGR),
    )

    scored_payload: list[dict[str, Any]] = []
    for sp in result.scored:
        p = sp.proposal
        x, y, bw, bh = 0, 0, 0, 0
        if np.any(p.mask):
            ys, xs = np.where(p.mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            x, y = x0, y0
            bw, bh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)

        scored_payload.append(
            {
                "candidate_id": p.candidate_id,
                "source": p.source,
                "area": p.area,
                "bbox_xywh": [x, y, bw, bh],
                "accepted": p.candidate_id in acc_ids,
                "validity_score": round(sp.filtering_score.validity_score, 6),
                "rejection_likelihood": round(sp.filtering_score.rejection_likelihood, 6),
                "heuristic_breakdown": {
                    k: round(float(v), 6) for k, v in sp.filtering_score.heuristic_breakdown.items()
                },
            }
        )

    payload: dict[str, Any] = {
        "metadata": {
            "input_count": result.metadata.input_count,
            "accepted_count": result.metadata.accepted_count,
            "rejected_count": result.metadata.rejected_count,
            "all_rejected_fallback": result.metadata.all_rejected_fallback,
            "dedup_removed_count": result.metadata.dedup_removed_count,
            "post_dedup_count": result.metadata.post_dedup_count,
        },
        "proposals": scored_payload,
        "accepted_ids": sorted({p.candidate_id for p in result.accepted}),
        "rejected_ids": sorted({p.candidate_id for p in result.rejected}),
    }
    json_path = os.path.join(root, f"{stem}_filtering_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
