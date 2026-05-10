"""Tunable compose-phase parameters (env + defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ComposeConfig:
    canvas_width: int
    canvas_height: int
    """Target canvas size (pixels)."""

    occupancy_ratio: float
    """Max fraction of canvas (width & height) the scaled foreground may occupy."""

    anchor_x_ratio: float
    """Horizontal anchor for subject center (0–1); 0.5 = centered."""

    anchor_y_ratio: float
    """Vertical anchor for subject center (0–1); >0.5 sits slightly lower."""

    background_rgb: tuple[int, int, int]
    """Solid sRGB background (e.g. pure white)."""

    alpha_bbox_threshold: int
    """Alpha value above which pixels count as visible for bbox / crop."""

    debug_enabled: bool
    """Write optional artifacts under ``debug/compose/``."""

    jpeg_quality: int
    """JPEG quality when compose becomes the final raster for the runner."""

    debug_line_color: tuple[int, int, int]
    """BGR color for debug overlays (OpenCV convention)."""


def load_compose_config() -> ComposeConfig:
    """Load ``IMAGE_UTIL_COMPOSE_*`` env vars."""

    br = _int_env("IMAGE_UTIL_COMPOSE_BG_R", 255)
    bg = _int_env("IMAGE_UTIL_COMPOSE_BG_G", 255)
    bb = _int_env("IMAGE_UTIL_COMPOSE_BG_B", 255)
    dr = _int_env("IMAGE_UTIL_COMPOSE_DEBUG_LINE_R", 0)
    dg = _int_env("IMAGE_UTIL_COMPOSE_DEBUG_LINE_G", 200)
    db = _int_env("IMAGE_UTIL_COMPOSE_DEBUG_LINE_B", 255)

    return ComposeConfig(
        canvas_width=_int_env("IMAGE_UTIL_COMPOSE_CANVAS_W", 2000),
        canvas_height=_int_env("IMAGE_UTIL_COMPOSE_CANVAS_H", 2000),
        occupancy_ratio=_float_env("IMAGE_UTIL_COMPOSE_OCCUPANCY", 0.82),
        anchor_x_ratio=_float_env("IMAGE_UTIL_COMPOSE_ANCHOR_X", 0.5),
        anchor_y_ratio=_float_env("IMAGE_UTIL_COMPOSE_ANCHOR_Y", 0.53),
        background_rgb=(br, bg, bb),
        alpha_bbox_threshold=_int_env("IMAGE_UTIL_COMPOSE_ALPHA_THRESH", 8),
        debug_enabled=_bool_env("IMAGE_UTIL_COMPOSE_DEBUG"),
        jpeg_quality=_int_env("IMAGE_UTIL_COMPOSE_JPEG_QUALITY", 94),
        debug_line_color=(db, dg, dr),
    )
