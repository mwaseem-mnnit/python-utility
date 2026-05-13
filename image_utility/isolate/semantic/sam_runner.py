"""Lazy MobileSAM (or compatible) mask generation — isolated from pipeline core."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch  # noqa: F401
import numpy as np
from numpy.typing import NDArray

from ..config import IsolateConfig

LOGGER = logging.getLogger(__name__)

_mask_generator_singleton: Any = None
_sam_init_error: str | None = None


def is_sam_available(cfg: IsolateConfig) -> bool:
    if not cfg.semantic_refinement_enabled:
        return False
    if not cfg.semantic_sam_checkpoint:
        return False
    if not Path(cfg.semantic_sam_checkpoint).is_file():
        return False
    try:
        import torch  # noqa: F401
        from mobile_sam import SamAutomaticMaskGenerator  # noqa: F401
        from mobile_sam import sam_model_registry  # noqa: F401
    except ImportError:
        return False
    return True


def _build_mask_generator(cfg: IsolateConfig):
    global _mask_generator_singleton, _sam_init_error
    if _mask_generator_singleton is not None:
        return _mask_generator_singleton
    if _sam_init_error is not None:
        return None

    try:
        import torch
        from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as e:
        _sam_init_error = str(e)
        LOGGER.debug("MobileSAM import failed: %s", e)
        return None

    ckpt = cfg.semantic_sam_checkpoint
    if not ckpt or not Path(ckpt).is_file():
        _sam_init_error = "checkpoint missing"
        return None

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.semantic_sam_use_gpu else "cpu")
    try:
        sam = sam_model_registry["vit_t"](checkpoint=ckpt)
        sam.to(device=device)
        sam.eval()
    except Exception as e:
        _sam_init_error = str(e)
        LOGGER.warning("[isolate] SAM model load failed: %s", e)
        return None

    try:
        _mask_generator_singleton = SamAutomaticMaskGenerator(
            sam,
            points_per_side=cfg.semantic_sam_points_per_side,
            pred_iou_thresh=cfg.semantic_sam_pred_iou_thresh,
            stability_score_thresh=cfg.semantic_sam_stability_thresh,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=cfg.semantic_sam_min_mask_area,
        )
    except TypeError:
        _mask_generator_singleton = SamAutomaticMaskGenerator(
            sam,
            points_per_side=cfg.semantic_sam_points_per_side,
        )
    except Exception as e:
        _sam_init_error = str(e)
        LOGGER.warning("[isolate] SAM mask generator init failed: %s", e)
        return None

    return _mask_generator_singleton


def generate_raw_masks(rgb: NDArray[np.uint8], cfg: IsolateConfig) -> list[dict] | None:
    """
    Run automatic mask generation on ``rgb`` (H×W×3, RGB uint8).

    Returns SAM mask dicts or ``None`` if inference is unavailable or fails.
    """
    gen = _build_mask_generator(cfg)
    if gen is None:
        return None

    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8 RGB")

    try:
        masks = gen.generate(rgb)
    except Exception as e:
        LOGGER.warning("[isolate] SAM inference failed: %s", e)
        return None

    if not masks:
        return None
    return list(masks)
