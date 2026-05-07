"""Shared configuration constants for image utilities."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "log"

# Env key in ``image_utility/.env``; value is ``compress`` or ``enhance``.
ENV_DEFAULT_JOB = "IMAGE_UTIL_DEFAULT_JOB"
DEFAULT_JOB_NAME = "enhance"

JOB_COMPRESS = "compress"
JOB_ENHANCE = "enhance"

# Relative to ``WORKSPACE_ROOT`` when not absolute (see ``IMAGE_UTIL_LOG_DIR`` in `.env`).
ENV_LOG_DIR = "IMAGE_UTIL_LOG_DIR"

