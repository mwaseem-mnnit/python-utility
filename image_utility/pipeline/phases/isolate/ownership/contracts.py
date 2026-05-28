"""Ownership stage DTOs — product vs support-object semantic reasoning.
Stage receives grouped regions from the grouping stage and produces ownership-labelled
groups with optionally clipped support sub-regions, ready for suppression.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray
BoolMask = NDArray[np.bool_]
UInt8 = NDArray[np.uint8]
# Canonical ownership labels assigned per-group; forwarded to suppression via
# affinity_breakdown metadata so suppression can weigh its own decisions.
OwnershipLabel = Literal["product", "support_object", "packaging", "environment", "uncertain"]
# ---------------------------------------------------------------------------
# Upstream context snapshots (numeric only; no cross-stage type leakage)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnershipRankingSnapshot:
    """Numeric summary of the upstream ranking pass."""
    candidate_count: int
    ambiguity_detected: bool
    top_confidence: float
    second_confidence: float
    confidence_separation: float
@dataclass(frozen=True)
class OwnershipGroupingSnapshot:
    """Numeric summary of the upstream grouping pass."""
    group_count: int
    grouping_ambiguity: bool
    top_group_confidence: float
    second_group_confidence: float
# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnershipGroupedRegionInput:
    """One grouped region entering ownership (copied at boundary; no cross-stage imports)."""
    group_id: int
    member_candidate_ids: tuple[int, ...]
    grouped_mask: BoolMask
    group_confidence: float
    affinity_breakdown: dict[str, float]
    geometry_metadata: dict[str, float | int]
@dataclass(frozen=True)
class OwnershipInput:
    regions: tuple[OwnershipGroupedRegionInput, ...]
    image_hw: tuple[int, int]
    ranking_snapshot: OwnershipRankingSnapshot | None = None
    grouping_snapshot: OwnershipGroupingSnapshot | None = None
# ---------------------------------------------------------------------------
# Per-group feature vector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnershipFeatures:
    """Ownership-specific geometry descriptors derived from the group mask."""
    # Basic geometry
    pixel_area: int
    relative_area: float            # fraction of total image area
    centroid_xy: tuple[float, float]
    center_distance_norm: float     # distance from image centre / image diagonal
    border_contact_ratio: float     # fraction of mask pixels touching image border
    elongation: float               # max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
    solidity: float                 # pixel_area / convex_hull_area
    contour_complexity: float       # perimeter^2 / (4*pi*area); 1.0 = perfect circle
    bbox_fill_ratio: float          # pixel_area / (bbox_w * bbox_h)
    # Multi-blob fragmentation (finger/scatter indicators)
    primary_blob_coverage: float    # largest blob / total mask area
    secondary_blob_ratio: float     # second-largest blob / largest blob
    blob_count: int                 # number of disconnected blobs in mask
    # Thin-bridge analysis (narrow connection => being held)
    thin_bridge_score: float        # 0=no bridge; 1=strong narrow bridge detected
    bridge_erosion_radius: int      # smallest erosion radius that caused split (0=no split)
    # Organic-shape indicators (hand/finger pattern proxies)
    finger_like_ratio: float        # fraction of sub-blobs that are elongated+thin
    irregular_boundary_score: float # convex_hull_deficit / convex_hull_area
# ---------------------------------------------------------------------------
# Per-group score
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnershipScore:
    """Ownership classification evidence for one group."""
    group_id: int
    product_likelihood: float       # 0-1; high => ecommerce product
    support_likelihood: float       # 0-1; high => hand / stand / holder
    packaging_likelihood: float     # 0-1; high => box / wrapping adjacent to product
    environment_likelihood: float   # 0-1; high => table slab / background
    assigned_label: OwnershipLabel
    score_breakdown: dict[str, float]
    features: OwnershipFeatures
# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OwnedGroup:
    """Group that passed ownership reasoning (mask may be clipped of support pixels)."""
    group_id: int
    member_candidate_ids: tuple[int, ...]
    surviving_mask: BoolMask        # original mask minus removed support sub-regions
    original_mask: BoolMask         # unmodified mask from grouping (for debug)
    removed_support_mask: BoolMask  # pixels classified as support; may be all-zeros
    ownership_label: OwnershipLabel
    product_likelihood: float
    support_likelihood: float
    ownership_confidence: float     # 1 - uncertainty; used by downstream suppression
    ownership_breakdown: dict[str, float]
    geometry_metadata: dict[str, float | int]
@dataclass(frozen=True)
class OwnershipMetadata:
    analyzed_group_count: int
    product_group_count: int
    support_group_count: int
    packaging_group_count: int
    environment_group_count: int
    uncertain_group_count: int
    global_ownership_confidence: float
    primary_product_group_id: int | None    # group_id of anchored primary product
    support_clipped_group_count: int        # groups from which support pixels were removed
@dataclass(frozen=True)
class OwnershipResult:
    owned_groups: tuple[OwnedGroup, ...]    # every group, all labels present
    scores: tuple[OwnershipScore, ...]
    combined_product_mask: BoolMask         # union of product + packaging surviving masks
    metadata: OwnershipMetadata
