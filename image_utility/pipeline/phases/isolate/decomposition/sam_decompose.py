"""MobileSAM mask generation — candidate proposals only."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import DecompositionConfig

LOGGER = logging.getLogger(__name__)
UInt8RGB = NDArray[np.uint8]


class SamDecomposer:
    """
    Lazily-initialized SAM automatic mask generator (instance-scoped, no module globals).
    """

    def __init__(self, cfg: DecompositionConfig) -> None:
        self._cfg = cfg
        self._generator: Any = None
        self._last_error: str | None = None

    def is_available(self) -> bool:
        if not self._cfg.sam_enabled:
            return False
        if not self._cfg.sam_checkpoint or not Path(self._cfg.sam_checkpoint).is_file():
            return False
        try:
            import torch  # noqa: F401
            from mobile_sam import SamAutomaticMaskGenerator  # noqa: F401
            from mobile_sam import sam_model_registry  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_generator(self) -> Any | None:
        if self._generator is not None:
            return self._generator
        if self._last_error is not None:
            return None
        try:
            import torch
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as e:
            self._last_error = str(e)
            return None

        ckpt = self._cfg.sam_checkpoint
        if not ckpt or not Path(ckpt).is_file():
            self._last_error = "missing_checkpoint"
            return None

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self._cfg.sam_use_gpu else "cpu"
        )
        try:
            sam = sam_model_registry["vit_t"](checkpoint=ckpt)
            sam.to(device=device)
            sam.eval()
        except Exception as e:
            self._last_error = str(e)
            LOGGER.warning("[decomposition] SAM model load failed: %s", e)
            return None

        try:
            self._generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self._cfg.sam_points_per_side,
                pred_iou_thresh=self._cfg.sam_pred_iou_thresh,
                stability_score_thresh=self._cfg.sam_stability_thresh,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=self._cfg.sam_min_mask_region_area,
            )
        except TypeError:
            self._generator = SamAutomaticMaskGenerator(
                sam,
                points_per_side=self._cfg.sam_points_per_side,
            )
        except Exception as e:
            self._last_error = str(e)
            LOGGER.warning("[decomposition] SAM generator init failed: %s", e)
            return None

        return self._generator

    def generate_raw(self, rgb: UInt8RGB) -> list[dict[str, Any]]:
        """Return SAM mask dicts (segmentation, predicted_iou, stability_score, …)."""
        gen = self._ensure_generator()
        if gen is None:
            return []
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            return []
        try:
            masks = gen.generate(rgb)
        except Exception as e:
            LOGGER.warning("[decomposition] SAM inference failed: %s", e)
            return []
        return list(masks) if masks else []
