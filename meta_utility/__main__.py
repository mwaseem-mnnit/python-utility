"""CLI entry point for meta utility jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from meta_utility.core.config import (
        DEFAULT_JOB_NAME,
        ENV_DEFAULT_JOB,
        JOB_WHATSAPP_MARKETING,
        load_meta_utility_env,
    )
    from meta_utility.dispatcher import run_job
else:
    from .core.config import DEFAULT_JOB_NAME, ENV_DEFAULT_JOB, JOB_WHATSAPP_MARKETING, load_meta_utility_env
    from .dispatcher import run_job


def main(argv: list[str] | None = None) -> int:
    load_meta_utility_env()

    parser = argparse.ArgumentParser(description="Run a meta utility job.")
    parser.add_argument(
        "job",
        nargs="?",
        help=(
            f"Job to run ({JOB_WHATSAPP_MARKETING}). "
            f"If omitted, reads {ENV_DEFAULT_JOB} from meta_utility/.env "
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

