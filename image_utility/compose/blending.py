"""Alpha compositing onto a solid-color canvas."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

UInt8RGBA = NDArray[np.uint8]
UInt8RGB = NDArray[np.uint8]


def blend_rgba_on_canvas(
    foreground_rgba: UInt8RGBA,
    canvas_height: int,
    canvas_width: int,
    origin_xy: tuple[int, int],
    background_rgb: tuple[int, int, int],
) -> UInt8RGB:
    """
    Alpha-blend ``foreground_rgba`` onto an RGB canvas at ``origin_xy`` (top-left).

    Uses straight-alpha over solid background. Returns H×W×3 ``uint8`` RGB.
    """
    fh, fw = foreground_rgba.shape[:2]
    if foreground_rgba.shape[2] != 4:
        raise OSError("foreground must be RGBA")

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    br, bg, bb = background_rgb
    canvas[:, :, 0] = br
    canvas[:, :, 1] = bg
    canvas[:, :, 2] = bb

    x0, y0 = origin_xy
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(canvas_height, y0 + fh)
    dst_x1 = min(canvas_width, x0 + fw)

    src_y0 = dst_y0 - y0
    src_x0 = dst_x0 - x0
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    src_x1 = src_x0 + (dst_x1 - dst_x0)

    if dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
        raise OSError("foreground does not overlap canvas")

    fg = foreground_rgba[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    alpha = fg[:, :, 3:4] / 255.0
    rgb = fg[:, :, :3]
    dst = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
    dst[:] = rgb * alpha + dst * (1.0 - alpha)

    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)
