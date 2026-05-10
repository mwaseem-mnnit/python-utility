"""Optional polish diagnostics under ``debug/polish/``."""

from __future__ import annotations

import os

import cv2
import numpy as np
from numpy.typing import NDArray

from image_utility.config import WORKSPACE_ROOT

from .config import PolishConfig

UInt8RGB = NDArray[np.uint8]


def polish_debug_dir() -> str:
    return str(WORKSPACE_ROOT / "debug" / "polish")


def _ensure() -> None:
    os.makedirs(polish_debug_dir(), exist_ok=True)


def write_polish_debug(
    cfg: PolishConfig,
    *,
    stem: str,
    before: UInt8RGB,
    after_sharpen: UInt8RGB | None,
    after_contrast: UInt8RGB | None,
    final: UInt8RGB,
) -> None:
    if not cfg.debug_enabled:
        return
    _ensure()
    root = polish_debug_dir()

    def _save(name: str, arr: UInt8RGB) -> None:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(root, f"{stem}_{name}.png"), bgr)

    _save("before", before)
    if after_sharpen is not None:
        _save("after_sharpen", after_sharpen)
    if after_contrast is not None:
        _save("after_contrast", after_contrast)
    _save("after", final)

    # Simple horizontal before | after strip
    h, w, _ = before.shape
    scale_h = min(800, h)
    scale_w = int(w * scale_h / h)
    b_s = cv2.resize(cv2.cvtColor(before, cv2.COLOR_RGB2BGR), (scale_w, scale_h))
    a_s = cv2.resize(cv2.cvtColor(final, cv2.COLOR_RGB2BGR), (scale_w, scale_h))
    strip = np.hstack([b_s, a_s])
    cv2.imwrite(os.path.join(root, f"{stem}_compare.png"), strip)
