"""JPEG → WebP conversion helpers (usable from scrapers and CLI)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def convert_to_webp(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
) -> None:
    """Resize a JPEG to ``(width, height)`` and save as WebP (quality 100)."""
    with Image.open(input_path) as img:
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "webp", quality=100)
        print(f"Success: {output_path} created at {width}x{height}")
