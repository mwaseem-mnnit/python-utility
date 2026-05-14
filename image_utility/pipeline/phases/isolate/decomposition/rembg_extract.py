"""rembg foreground bootstrap — alpha only; no region decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from image_utility.pipeline.phases.isolate.decomposition.segmentation import extract_alpha, rgba_from_rgb

from .config import DecompositionConfig

UInt8RGB = NDArray[np.uint8]
UInt8RGBA = NDArray[np.uint8]
UInt8Alpha = NDArray[np.uint8]


@dataclass(frozen=True)
class RembgExtractOutput:
    rgba: UInt8RGBA
    alpha: UInt8Alpha


def extract_foreground(rgb: UInt8RGB, cfg: DecompositionConfig) -> RembgExtractOutput:
    """Run rembg; preserve soft edges from model output."""
    rgba = rgba_from_rgb(rgb, model_name=cfg.rembg_model_name)
    alpha = extract_alpha(rgba)
    return RembgExtractOutput(rgba=np.ascontiguousarray(rgba), alpha=np.ascontiguousarray(alpha))
