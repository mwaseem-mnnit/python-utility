"""rembg-based segmentation → RGBA (replaceable backend)."""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from rembg import remove

LOGGER = logging.getLogger(__name__)

_SESSION = None
_SESSION_MODEL: str | None = None


def get_rembg_session(model_name: str | None = None):
    """Single process-wide rembg session; optional model override from config."""
    global _SESSION, _SESSION_MODEL
    from rembg import new_session

    if _SESSION is None or (model_name or None) != _SESSION_MODEL:
        _SESSION = new_session(model_name) if model_name else new_session()
        _SESSION_MODEL = model_name
    return _SESSION


def rgba_from_rgb(rgb: np.ndarray, *, model_name: str | None = None) -> np.ndarray:
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
        LOGGER.warning("rembg failed: %s", exc)
        raise OSError(f"rembg segmentation failed: {exc}") from exc

    if rgba_pil.mode != "RGBA":
        rgba_pil = rgba_pil.convert("RGBA")
    return np.asarray(rgba_pil, dtype=np.uint8)


def alpha_channel(rgba: np.ndarray) -> np.ndarray:
    """Return alpha as H×W uint8 copy."""
    if rgba.shape[2] != 4:
        raise OSError("expected RGBA array")
    return rgba[:, :, 3].copy()
