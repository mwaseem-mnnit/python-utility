"""rembg integration and alpha extraction (replaceable segmentation backend)."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rembg import remove

from image_utility.isolate.config import IsolateConfig

LOGGER = logging.getLogger(__name__)

_SESSION = None
_SESSION_MODEL: str | None = None

UInt8RGBA = NDArray[np.uint8]
UInt8RGB = NDArray[np.uint8]
UInt8Alpha = NDArray[np.uint8]


def get_rembg_session(model_name: str | None = None):
    """Single process-wide rembg session; optional model override from config."""
    global _SESSION, _SESSION_MODEL
    from rembg import new_session

    if _SESSION is None or (model_name or None) != _SESSION_MODEL:
        _SESSION = new_session(model_name) if model_name else new_session()
        _SESSION_MODEL = model_name
    return _SESSION


def rgba_from_rgb(
    rgb: UInt8RGB,
    *,
    model_name: str | None = None,
) -> UInt8RGBA:
    """
    Run rembg on ``rgb`` (H×W×3 RGB uint8) and return H×W×4 RGBA uint8.

    Preserves soft edges from the model output; failures raise ``OSError``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise OSError("segmentation expects RGB array H×W×3")
    h, w = rgb.shape[:2]
    if h < 2 or w < 2:
        raise OSError("image too small for segmentation")

    session = get_rembg_session(model_name)
    try:
        pil_in = Image.fromarray(rgb, "RGB")
        rgba_pil = remove(pil_in, session=session)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("[isolate] rembg failed: %s", exc)
        raise OSError(f"rembg segmentation failed: {exc}") from exc

    if rgba_pil.mode != "RGBA":
        rgba_pil = rgba_pil.convert("RGBA")
    return np.asarray(rgba_pil, dtype=np.uint8)


def segment_rgba(rgb: UInt8RGB, cfg: IsolateConfig) -> UInt8RGBA:
    """Segment ``rgb`` with rembg using ``cfg`` model settings."""
    return rgba_from_rgb(rgb, model_name=cfg.rembg_model_name)


def extract_alpha(rgba: UInt8RGBA) -> UInt8Alpha:
    """Return H×W alpha channel (uint8 copy)."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise OSError("expected RGBA array")
    return rgba[:, :, 3].copy()
