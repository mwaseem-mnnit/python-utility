"""Conservative candidate normalization (recall-first)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import DecompositionConfig
from .contracts import BoolMask, SemanticMaskCandidate

UInt8RGB = NDArray[np.uint8]


def _iou(a: BoolMask, b: BoolMask) -> float:
    inter = np.count_nonzero(np.logical_and(a, b))
    union = np.count_nonzero(np.logical_or(a, b))
    return float(inter) / float(max(union, 1))


def raw_dicts_to_candidates(
    raw: list[dict],
    image_hw: tuple[int, int],
    cfg: DecompositionConfig,
) -> list[SemanticMaskCandidate]:
    ih, iw = image_hw
    candidates: list[SemanticMaskCandidate] = []
    for idx, entry in enumerate(raw):
        seg = entry.get("segmentation")
        if seg is None:
            continue
        m = np.asarray(seg, dtype=bool)
        if m.shape != (ih, iw):
            m = (
                cv2.resize(
                    m.astype(np.uint8),
                    (iw, ih),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )
        area = int(np.count_nonzero(m))
        if area < cfg.normalize_min_mask_area:
            continue
        candidates.append(
            SemanticMaskCandidate(
                candidate_id=idx,
                mask=m,
                source="sam",
                predicted_iou=float(entry.get("predicted_iou", 0.0)),
                stability_score=float(entry.get("stability_score", 0.0)),
                area=area,
            )
        )
    return candidates


def connected_regions_to_candidates(
    labels: NDArray[np.int32],
    cfg: DecompositionConfig,
) -> list[SemanticMaskCandidate]:
    """One bool mask per CC label (proposal recall when SAM is off or as complement)."""
    mx = int(labels.max())
    if mx < 1:
        return []
    out: list[SemanticMaskCandidate] = []
    cid = 0
    for lab in range(1, mx + 1):
        m = labels == lab
        area = int(np.count_nonzero(m))
        if area < cfg.normalize_min_mask_area:
            continue
        out.append(
            SemanticMaskCandidate(
                candidate_id=cid,
                mask=m,
                source="connected_component",
                area=area,
            )
        )
        cid += 1
    return out


def renumber_candidates(cands: list[SemanticMaskCandidate]) -> list[SemanticMaskCandidate]:
    return [
        SemanticMaskCandidate(
            candidate_id=i,
            mask=c.mask,
            source=c.source,
            predicted_iou=c.predicted_iou,
            stability_score=c.stability_score,
            area=c.area,
        )
        for i, c in enumerate(cands)
    ]


def deduplicate_conservative(
    candidates: list[SemanticMaskCandidate],
    cfg: DecompositionConfig,
) -> list[SemanticMaskCandidate]:
    """Drop near-duplicate masks (very high IoU); keep higher-quality / larger-first ordering."""
    if len(candidates) <= 1:
        return renumber_candidates(candidates)

    scored = sorted(
        candidates,
        key=lambda c: (
            c.predicted_iou + 0.5 * c.stability_score,
            c.area,
        ),
        reverse=True,
    )
    kept: list[SemanticMaskCandidate] = []
    for cand in scored:
        if any(
            _iou(cand.mask, k.mask) >= cfg.candidate_overlap_dedup_threshold for k in kept
        ):
            continue
        kept.append(cand)

    kept.sort(key=lambda c: c.area, reverse=True)
    if len(kept) > cfg.candidate_max_regions:
        kept = kept[: cfg.candidate_max_regions]

    return renumber_candidates(kept)


def morph_post_open_binary(mask_255: NDArray[np.uint8], ksize: int) -> NDArray[np.uint8]:
    if ksize <= 0:
        return mask_255
    k = ksize | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask_255, cv2.MORPH_OPEN, ker)

