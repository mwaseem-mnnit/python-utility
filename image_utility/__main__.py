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
    from image_utility.config import DEFAULT_JOB_ENV, JOB_COMPRESS, JOB_ENHANCE
    from image_utility.dispatcher import run_job
else:
    from .config import DEFAULT_JOB_ENV, JOB_COMPRESS, JOB_ENHANCE
    from .dispatcher import run_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an image utility job.")
    parser.add_argument(
        "job",
        nargs="?",
        help=f"Job to run ({JOB_COMPRESS}, {JOB_ENHANCE}). Falls back to {DEFAULT_JOB_ENV}.",
    )
    args = parser.parse_args(argv)

    job = args.job or os.getenv(DEFAULT_JOB_ENV, "").strip()
    if not job:
        parser.error(f"provide a job or set {DEFAULT_JOB_ENV}")
    return run_job(job)


if __name__ == "__main__":
    raise SystemExit(main())

