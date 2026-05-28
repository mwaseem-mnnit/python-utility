"""Ownership stage orchestration — product anchor, scoring, and support clipping."""
from __future__ import annotations
import logging
import math
import numpy as np
from .clipping import clip_support_subregion
from .config import OwnershipConfig, load_ownership_config
from .contracts import (
    OwnedGroup,
    OwnershipGroupedRegionInput,
    OwnershipInput,
    OwnershipLabel,
    OwnershipMetadata,
    OwnershipResult,
    OwnershipScore,
)
from .debug.writers import write_ownership_debug
from .features import extract_ownership_features
from .scoring import score_all_groups
LOGGER = logging.getLogger(__name__)
def _identify_anchor(
    regions: tuple[OwnershipGroupedRegionInput, ...],
    image_hw: tuple[int, int],
    cfg: OwnershipConfig,
) -> int | None:
    """
    Select the primary product anchor group: the group most likely to be the
    core ecommerce product.  Uses a weighted combination of group confidence,
    centrality, and relative area — entirely deterministic.
    """
    if not regions:
        return None
    h, w = image_hw
    diag = float(math.hypot(w, h)) + cfg.math_epsilon
    image_area = float(max(h * w, 1))
    best_id: int | None = None
    best_score = -1.0
    for r in regions:
        if r.group_confidence < cfg.product_min_ranking_confidence:
            continue
        gm = r.geometry_metadata
        cx = float(gm.get("centroid_x", w / 2.0))
        cy = float(gm.get("centroid_y", h / 2.0))
        dist = math.hypot(cx - w / 2.0, cy - h / 2.0) / diag
        centre_score = float(max(0.0, 1.0 - dist * 2.0))
        area = int(gm.get("pixel_area", 0))
        area_score = float(min(1.0, area / max(image_area * 0.5, 1.0)))
        score = (
            cfg.anchor_confidence_weight * r.group_confidence
            + cfg.anchor_centre_weight * centre_score
            + cfg.anchor_area_weight * area_score
        )
        if score > best_score:
            best_score = score
            best_id = r.group_id
    if best_id is None and regions:
        # Fallback: highest confidence group regardless of threshold
        best_id = max(regions, key=lambda r: r.group_confidence).group_id
    return best_id
def _clip_if_needed(
    region: OwnershipGroupedRegionInput,
    score: OwnershipScore,
    image_hw: tuple[int, int],
    cfg: OwnershipConfig,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Run thin-bridge clipping when the group carries material support evidence
    (bridge score OR finger pattern) but also passes a minimum product area guard.
    Returns (surviving_mask, removed_support_mask, was_clipped).
    """
    should_clip = (
        score.features.thin_bridge_score >= 0.25
        or (
            score.features.finger_like_ratio >= cfg.support_min_finger_like_ratio
            and score.features.secondary_blob_ratio >= cfg.support_min_secondary_blob_ratio
        )
    )
    if not should_clip:
        return (
            np.ascontiguousarray(region.grouped_mask.copy()),
            np.zeros_like(region.grouped_mask),
            False,
        )
    product_mask, support_mask, clipped = clip_support_subregion(
        region.grouped_mask, image_hw, cfg
    )
    return product_mask, support_mask, clipped
class OwnershipProcessor:
    """
    Semantic product ownership — classifies each group as product / support_object /
    packaging / environment / uncertain and removes support sub-regions via
    thin-bridge erosion.
    """
    def __init__(self, cfg: OwnershipConfig | None = None) -> None:
        self._cfg = cfg or load_ownership_config()
    @property
    def config(self) -> OwnershipConfig:
        return self._cfg
    def run(
        self,
        inp: OwnershipInput,
        rgb: np.ndarray,
        stem: str,
    ) -> OwnershipResult:
        if not inp.regions:
            raise ValueError("ownership stage requires at least one grouped region")
        cfg = self._cfg
        LOGGER.info("[ownership] analyzing grouped regions=%d", len(inp.regions))
        # Step 1: identify primary product anchor
        anchor_id = _identify_anchor(inp.regions, inp.image_hw, cfg)
        LOGGER.info("[ownership] anchor group_id=%s", anchor_id)
        # Step 2: extract ownership features for every group
        all_features = tuple(
            extract_ownership_features(r, inp.image_hw, cfg) for r in inp.regions
        )
        # Step 3: score each group (four likelihoods)
        scores = score_all_groups(inp.regions, all_features, cfg, anchor_id)
        score_by_id = {s.group_id: s for s in scores}
        # Step 4: optionally clip support sub-regions, build OwnedGroup list
        owned_groups: list[OwnedGroup] = []
        clipped_count = 0
        for region in sorted(inp.regions, key=lambda r: r.group_id):
            score = score_by_id[region.group_id]
            surviving, removed, was_clipped = _clip_if_needed(
                region, score, inp.image_hw, cfg
            )
            if was_clipped:
                clipped_count += 1
                LOGGER.info(
                    "[ownership] clipped support sub-region from group_id=%d "
                    "(removed_px=%d)",
                    region.group_id,
                    int(np.count_nonzero(removed)),
                )
            label: OwnershipLabel = score.assigned_label
            own_conf = float(max(
                cfg.confidence_floor,
                min(1.0, score.product_likelihood * (1.0 - score.support_likelihood * 0.8)),
            ))
            # Pack ownership breakdown into the group (suppression can read it)
            ownership_breakdown = dict(score.score_breakdown)
            ownership_breakdown["ownership_label"] = {
                "product": 1.0, "support_object": 2.0,
                "packaging": 3.0, "environment": 4.0, "uncertain": 5.0,
            }.get(label, 5.0)
            ownership_breakdown["ownership_confidence"] = own_conf
            gm = dict(region.geometry_metadata)
            gm["ownership_label_id"] = ownership_breakdown["ownership_label"]
            gm["ownership_confidence"] = own_conf
            owned_groups.append(
                OwnedGroup(
                    group_id=region.group_id,
                    member_candidate_ids=region.member_candidate_ids,
                    surviving_mask=surviving,
                    original_mask=np.ascontiguousarray(region.grouped_mask.copy()),
                    removed_support_mask=removed,
                    ownership_label=label,
                    product_likelihood=score.product_likelihood,
                    support_likelihood=score.support_likelihood,
                    ownership_confidence=own_conf,
                    ownership_breakdown=ownership_breakdown,
                    geometry_metadata=gm,
                )
            )
        # Step 5: aggregate counts
        label_counts: dict[str, int] = {
            "product": 0, "support_object": 0, "packaging": 0,
            "environment": 0, "uncertain": 0,
        }
        for g in owned_groups:
            label_counts[g.ownership_label] += 1
        for label_key, count in label_counts.items():
            LOGGER.info("[ownership] label=%s count=%d", label_key, count)
        # Step 6: build combined product mask (product + packaging)
        product_mask = np.zeros(inp.regions[0].grouped_mask.shape, dtype=bool)
        for g in owned_groups:
            if g.ownership_label in ("product", "packaging", "uncertain"):
                product_mask = np.logical_or(product_mask, g.surviving_mask)
        # Step 7: area-weighted global ownership confidence
        total_px = sum(int(np.count_nonzero(g.surviving_mask)) for g in owned_groups)
        if total_px > 0:
            g_conf = sum(
                int(np.count_nonzero(g.surviving_mask)) * g.ownership_confidence
                for g in owned_groups
            ) / total_px
            g_conf = float(max(cfg.confidence_floor, min(1.0, g_conf)))
        else:
            g_conf = float(cfg.confidence_floor)
        LOGGER.info("[ownership] global_ownership_confidence=%.3f", g_conf)
        meta = OwnershipMetadata(
            analyzed_group_count=len(inp.regions),
            product_group_count=label_counts["product"],
            support_group_count=label_counts["support_object"],
            packaging_group_count=label_counts["packaging"],
            environment_group_count=label_counts["environment"],
            uncertain_group_count=label_counts["uncertain"],
            global_ownership_confidence=g_conf,
            primary_product_group_id=anchor_id,
            support_clipped_group_count=clipped_count,
        )
        result = OwnershipResult(
            owned_groups=tuple(owned_groups),
            scores=scores,
            combined_product_mask=np.ascontiguousarray(product_mask),
            metadata=meta,
        )
        write_ownership_debug(cfg, stem=stem, rgb=rgb, inp=inp, result=result)
        return result
