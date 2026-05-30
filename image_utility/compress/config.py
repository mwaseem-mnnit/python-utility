"""Export/compress phase tunables (env + defaults)."""

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
class CompressConfig:
    """Multi-format export configuration for ecommerce delivery."""

    webp_size: int
    """Square dimension for WebP listing image (px)."""

    webp_quality: int
    """WebP lossy quality (0–100; 100 = lossless-ish)."""

    jpeg_size: int
    """Square dimension for JPEG full-res export (px); 0 = use source size."""

    jpeg_quality: int
    """JPEG quality for full-res export."""

    thumbnail_size: int
    """Square dimension for thumbnail (px)."""

    thumbnail_quality: int
    """WebP quality for thumbnail."""

    emit_webp: bool
    """Generate WebP listing image."""

    emit_jpeg: bool
    """Generate JPEG full-res."""

    emit_thumbnail: bool
    """Generate thumbnail WebP."""

    is_thumbnail_only: bool
    """When True, only emit thumbnail (thumbnail-only batch mode)."""


def load_compress_config() -> CompressConfig:
    """Load ``IMAGE_UTIL_COMPRESS_*`` env vars."""
    is_thumb = os.getenv("IMAGE_UTIL_THUMBNAIL", "").strip() == "1"

    return CompressConfig(
        webp_size=_int_env("IMAGE_UTIL_COMPRESS_WEBP_SIZE", 950),
        webp_quality=_int_env("IMAGE_UTIL_COMPRESS_WEBP_QUALITY", 92),
        jpeg_size=_int_env("IMAGE_UTIL_COMPRESS_JPEG_SIZE", 0),
        jpeg_quality=_int_env("IMAGE_UTIL_COMPRESS_JPEG_QUALITY", 94),
        thumbnail_size=_int_env("IMAGE_UTIL_COMPRESS_THUMB_SIZE", 420),
        thumbnail_quality=_int_env("IMAGE_UTIL_COMPRESS_THUMB_QUALITY", 82),
        emit_webp=not _bool_env("IMAGE_UTIL_COMPRESS_NO_WEBP"),
        emit_jpeg=not _bool_env("IMAGE_UTIL_COMPRESS_NO_JPEG"),
        emit_thumbnail=_bool_env("IMAGE_UTIL_COMPRESS_THUMBNAIL") or is_thumb,
        is_thumbnail_only=is_thumb,
    )

