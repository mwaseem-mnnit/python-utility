"""Stage-local ownership configuration (env prefix IMAGE_UTIL_ISOLATE_OWN_)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default
def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
def _bool_env(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in ("1", "true", "yes", "on")
def _intlist_env(key: str, default: list[int]) -> list[int]:
    """Parse comma-separated integers from env (e.g. '3,5,7')."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        return default
@dataclass(frozen=True)
class OwnershipConfig:
    # -----------------------------------------------------------------------
    # Thin-bridge erosion (support sub-region clipping within a grouped mask)
    # -----------------------------------------------------------------------
    bridge_erosion_radii: tuple[int, ...]
    """Ascending erosion radii to probe for thin bridges (e.g. 3, 5, 7, 9).
    Smaller radius = finer bridge detection; larger = coarser / more aggressive."""
    bridge_min_split_size: int
    """Minimum pixel area a sub-region must have after erosion to be considered a
    meaningful blob (not noise)."""
    bridge_periphery_threshold: float
    """Normalised centre-distance above which a post-erosion sub-blob is considered
    peripheral (likely support).  0=strict centre; 1.0=full radius."""
    bridge_finger_elongation: float
    """Elongation threshold for a sub-blob to be labelled 'finger-like' after erosion."""
    bridge_product_min_area_ratio: float
    """After a bridge split, the product blob must cover at least this fraction of the
    original mask; otherwise the split is rejected (avoids destroying small products)."""
    # -----------------------------------------------------------------------
    # Support-object detection (standalone group heuristics)
    # -----------------------------------------------------------------------
    support_min_elongation: float
    """Elongation threshold above which a group is penalised toward support_object."""
    support_max_solidity: float
    """Solidity below which a group is penalised toward support_object (organic shape)."""
    support_min_contour_complexity: float
    """Normalised complexity above which organic/ragged boundary is suspected."""
    support_min_secondary_blob_ratio: float
    """If secondary_blob_ratio >= this value, scattered finger-like geometry is suspected."""
    support_min_finger_like_ratio: float
    """Fraction of elongated sub-blobs needed to activate finger-scatter penalty."""
    # -----------------------------------------------------------------------
    # Environment-slab detection
    # -----------------------------------------------------------------------
    env_max_image_ratio: float
    """Relative area >= this flags a group as background/table slab."""
    env_min_border_contact: float
    """Border contact ratio >= this adds environment evidence."""
    # -----------------------------------------------------------------------
    # Packaging detection
    # -----------------------------------------------------------------------
    pkg_min_bbox_fill: float
    """Rectangular bbox fill >= this suggests box/packaging."""
    pkg_max_elongation: float
    """Packaging is compact; elongation above this disqualifies packaging label."""
    # -----------------------------------------------------------------------
    # Product anchor identification
    # -----------------------------------------------------------------------
    product_min_ranking_confidence: float
    """Group confidence below this disqualifies primary product anchor role."""
    anchor_centre_weight: float
    """Weight for centrality in anchor scoring (higher = prefer central groups)."""
    anchor_confidence_weight: float
    """Weight for group_confidence in anchor scoring."""
    anchor_area_weight: float
    """Weight for relative area in anchor scoring (prefer prominent groups)."""
    # -----------------------------------------------------------------------
    # Label assignment thresholds
    # -----------------------------------------------------------------------
    product_label_threshold: float
    """product_likelihood must exceed this to assign 'product' label."""
    support_label_threshold: float
    """support_likelihood must exceed this to assign 'support_object' label."""
    environment_label_threshold: float
    """environment_likelihood must exceed this to assign 'environment' label."""
    packaging_label_threshold: float
    """packaging_likelihood must exceed this to assign 'packaging' label."""
    # -----------------------------------------------------------------------
    # Scoring weights
    # -----------------------------------------------------------------------
    w_product_centre: float
    w_product_confidence: float
    w_product_solidity: float
    w_product_compactness: float
    w_support_elongation: float
    w_support_low_solidity: float
    w_support_complexity: float
    w_support_secondary_blob: float
    w_support_thin_bridge: float
    w_support_finger_like: float
    w_support_periphery: float
    w_env_area: float
    w_env_border: float
    w_pkg_bbox_fill: float
    w_pkg_compactness: float
    # -----------------------------------------------------------------------
    # General
    # -----------------------------------------------------------------------
    confidence_floor: float
    math_epsilon: float
    debug_enabled: bool
    debug_top_groups: int
def load_ownership_config() -> OwnershipConfig:
    """Load configuration from IMAGE_UTIL_ISOLATE_OWN_* env vars."""
    radii = _intlist_env("IMAGE_UTIL_ISOLATE_OWN_BRIDGE_RADII", [3, 5, 7, 9])
    return OwnershipConfig(
        bridge_erosion_radii=tuple(sorted(set(r for r in radii if r > 0))),
        bridge_min_split_size=_int_env("IMAGE_UTIL_ISOLATE_OWN_BRIDGE_MIN_SIZE", 80),
        bridge_periphery_threshold=_float_env("IMAGE_UTIL_ISOLATE_OWN_BRIDGE_PERIPH", 0.28),
        bridge_finger_elongation=_float_env("IMAGE_UTIL_ISOLATE_OWN_BRIDGE_FINGER_ELONG", 2.5),
        bridge_product_min_area_ratio=_float_env("IMAGE_UTIL_ISOLATE_OWN_BRIDGE_PROD_MIN", 0.30),
        support_min_elongation=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPPORT_ELONG", 3.0),
        support_max_solidity=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPPORT_SOLIDITY", 0.60),
        support_min_contour_complexity=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPPORT_COMPLEXITY", 4.0),
        support_min_secondary_blob_ratio=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPPORT_SEC_BLOB", 0.08),
        support_min_finger_like_ratio=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPPORT_FINGER", 0.30),
        env_max_image_ratio=_float_env("IMAGE_UTIL_ISOLATE_OWN_ENV_MAX_RATIO", 0.78),
        env_min_border_contact=_float_env("IMAGE_UTIL_ISOLATE_OWN_ENV_BORDER", 0.20),
        pkg_min_bbox_fill=_float_env("IMAGE_UTIL_ISOLATE_OWN_PKG_FILL", 0.65),
        pkg_max_elongation=_float_env("IMAGE_UTIL_ISOLATE_OWN_PKG_ELONG", 2.8),
        product_min_ranking_confidence=_float_env("IMAGE_UTIL_ISOLATE_OWN_PROD_CONF_MIN", 0.10),
        anchor_centre_weight=_float_env("IMAGE_UTIL_ISOLATE_OWN_ANCHOR_CENTRE", 1.20),
        anchor_confidence_weight=_float_env("IMAGE_UTIL_ISOLATE_OWN_ANCHOR_CONF", 1.40),
        anchor_area_weight=_float_env("IMAGE_UTIL_ISOLATE_OWN_ANCHOR_AREA", 0.80),
        product_label_threshold=_float_env("IMAGE_UTIL_ISOLATE_OWN_PROD_THRESH", 0.42),
        support_label_threshold=_float_env("IMAGE_UTIL_ISOLATE_OWN_SUPP_THRESH", 0.50),
        environment_label_threshold=_float_env("IMAGE_UTIL_ISOLATE_OWN_ENV_THRESH", 0.55),
        packaging_label_threshold=_float_env("IMAGE_UTIL_ISOLATE_OWN_PKG_THRESH", 0.45),
        w_product_centre=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PROD_CENTRE", 1.20),
        w_product_confidence=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PROD_CONF", 1.40),
        w_product_solidity=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PROD_SOLID", 0.90),
        w_product_compactness=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PROD_COMPACT", 0.70),
        w_support_elongation=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_ELONG", 1.10),
        w_support_low_solidity=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_SOLID", 0.80),
        w_support_complexity=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_COMPLEX", 0.75),
        w_support_secondary_blob=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_SEC", 0.85),
        w_support_thin_bridge=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_BRIDGE", 1.30),
        w_support_finger_like=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_FINGER", 1.00),
        w_support_periphery=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_SUPP_PERIPH", 0.65),
        w_env_area=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_ENV_AREA", 1.20),
        w_env_border=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_ENV_BORDER", 0.90),
        w_pkg_bbox_fill=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PKG_FILL", 1.10),
        w_pkg_compactness=_float_env("IMAGE_UTIL_ISOLATE_OWN_W_PKG_COMPACT", 0.80),
        confidence_floor=_float_env("IMAGE_UTIL_ISOLATE_OWN_CONF_FLOOR", 0.04),
        math_epsilon=_float_env("IMAGE_UTIL_ISOLATE_OWN_EPS", 1e-6),
        debug_enabled=_bool_env("IMAGE_UTIL_ISOLATE_DEBUG"),
        debug_top_groups=_int_env("IMAGE_UTIL_ISOLATE_OWN_DEBUG_TOP", 8),
    )
