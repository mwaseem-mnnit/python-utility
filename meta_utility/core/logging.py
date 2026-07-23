"""Logging helpers for meta utility jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app_logging import init_logging

from meta_utility.core.config import DEFAULT_LOG_DIR, ENV_LOG_DIR, PACKAGE_DIR


def utility_log_dir() -> Path:
    """Directory for logs: ``META_UTIL_LOG_DIR`` or workspace ``log/``."""
    raw = os.getenv(ENV_LOG_DIR, "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (PACKAGE_DIR.parent / path).resolve()
    return DEFAULT_LOG_DIR


def init_job_logging(log_filename: str) -> logging.Logger:
    """Configure shared logging for a meta utility job."""
    log_dir = utility_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=log_dir / log_filename, also_stdout=True)
    return logging.getLogger(__name__)

