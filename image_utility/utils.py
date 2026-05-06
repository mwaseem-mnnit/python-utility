"""Small shared helpers for image utility jobs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from app_logging import init_logging

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_job_env(job_dir: Path) -> None:
    """Load ``.env`` from a job directory."""
    load_dotenv(job_dir / ".env")


def init_job_logging(default_filename: str) -> logging.Logger:
    """Initialize common file/stdout logging and return the caller logger."""
    init_logging(also_stdout=True, default_filename=default_filename)
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

