"""Shared configuration constants for image utilities."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_JOB_ENV = "enhance"

JOB_COMPRESS = "compress"
JOB_ENHANCE = "enhance"

