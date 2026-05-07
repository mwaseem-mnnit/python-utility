"""Small shared helpers for image utility jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from app_logging import init_logging

from .config import DEFAULT_LOG_DIR, ENV_LOG_DIR, PACKAGE_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_image_utility_env() -> None:
    """Load ``image_utility/.env`` (single configuration for all jobs)."""
    load_dotenv(PACKAGE_DIR / ".env")


def utility_log_dir() -> Path:
    """Directory for log files: ``IMAGE_UTIL_LOG_DIR`` or ``<workspace>/log``."""
    raw = os.getenv(ENV_LOG_DIR, "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (PACKAGE_DIR.parent / p).resolve()
    return DEFAULT_LOG_DIR


def init_job_logging(log_filename: str) -> logging.Logger:
    """Configure :mod:`app_logging` under workspace ``log/`` plus stdout."""
    log_dir = utility_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=log_dir / log_filename, also_stdout=True)
    return logging.getLogger(__name__)


def resolve_dir_from_env(var_name: str) -> Path | None:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def parse_positive_int_env(var_name: str) -> int | None:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        return None
    return value if value > 0 else None


def sorted_image_files(directory: Path, *, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    allowed = {ext.lower() for ext in exts}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    )

