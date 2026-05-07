"""Unified entry point for image utility jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from image_utility.config import (
        DEFAULT_JOB_NAME,
        ENV_DEFAULT_JOB,
        JOB_COMPRESS,
        JOB_ENHANCE,
    )
    from image_utility.dispatcher import run_job
    from image_utility.utils import load_image_utility_env
else:
    from .config import DEFAULT_JOB_NAME, ENV_DEFAULT_JOB, JOB_COMPRESS, JOB_ENHANCE
    from .dispatcher import run_job
    from .utils import load_image_utility_env


def main(argv: list[str] | None = None) -> int:
    load_image_utility_env()

    parser = argparse.ArgumentParser(description="Run an image utility job.")
    parser.add_argument(
        "job",
        nargs="?",
        help=(
            f"Job to run ({JOB_COMPRESS}, {JOB_ENHANCE}). "
            f"If omitted, reads {ENV_DEFAULT_JOB} from image_utility/.env "
            f"(default name: {DEFAULT_JOB_NAME})."
        ),
    )
    args = parser.parse_args(argv)

    job = (args.job or os.getenv(ENV_DEFAULT_JOB, DEFAULT_JOB_NAME)).strip()
    if not job:
        parser.error(f"provide a job or set {ENV_DEFAULT_JOB}")
    return run_job(job)


if __name__ == "__main__":
    raise SystemExit(main())

