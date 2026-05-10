"""Optional shadow debug images under ``debug/shadow/``."""

from __future__ import annotations

import os

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .config import ShadowConfig

FloatMask = NDArray[np.float32]
UInt8RGB = NDArray[np.uint8]


def shadow_debug_dir() -> str:
    return str(WORKSPACE_ROOT / "debug" / "shadow")


def _ensure() -> None:
    os.makedirs(shadow_debug_dir(), exist_ok=True)


def write_shadow_debug(
    cfg: ShadowConfig,
    *,
    stem: str,
    raw_mask: FloatMask,
    blurred: FloatMask,
    rgb_after: UInt8RGB,
) -> None:
    if not cfg.debug_enabled:
        return
    _ensure()
    root = shadow_debug_dir()
    raw_u8 = np.clip(raw_mask * 255.0, 0, 255).astype(np.uint8)
    blur_u8 = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(root, f"{stem}_shadow_raw.png"), raw_u8)
    cv2.imwrite(os.path.join(root, f"{stem}_shadow_blur.png"), blur_u8)
    cv2.imwrite(
        os.path.join(root, f"{stem}_shadow_final.png"),
        cv2.cvtColor(rgb_after, cv2.COLOR_RGB2BGR),
    )
