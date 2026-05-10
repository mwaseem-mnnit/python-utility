"""Compress job package."""

from .phase import THUMBNAIL_SIZE, WEBP_SIZE, CompressPhase
from .processor import run

__all__ = ["run", "CompressPhase", "WEBP_SIZE", "THUMBNAIL_SIZE"]
