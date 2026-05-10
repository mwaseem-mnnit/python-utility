"""Shared configuration constants for image utilities."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "log"


ENV_INPUT_DIR = "IMAGE_UTIL_INPUT_DIR"
ENV_OUTPUT_DIR = "IMAGE_UTIL_OUTPUT_DIR"
ENV_MAX_FILES = "IMAGE_UTIL_MAX_FILES"
ENV_PIPELINE_STEPS = "IMAGE_UTIL_PIPELINE_STEPS"
ENV_THUMBNAIL = "IMAGE_UTIL_THUMBNAIL"


# Env key in ``image_utility/.env``; value is ``pipeline``, ``compress``, or ``enhance``.
ENV_DEFAULT_JOB = "pipeline"
DEFAULT_JOB_NAME = "pipeline"

JOB_COMPRESS = "compress"
JOB_PIPELINE = "pipeline"
JOB_ENHANCE = "legacy_enhance"

# Relative to ``WORKSPACE_ROOT`` when not absolute (see ``IMAGE_UTIL_LOG_DIR`` in `.env`).
ENV_LOG_DIR = "IMAGE_UTIL_LOG_DIR"

