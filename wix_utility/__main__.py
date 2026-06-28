"""CLI entry point for Wix utility jobs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wix_utility.core.config import (
        DEFAULT_JOB_NAME,
        ENV_DEFAULT_JOB,
        JOB_COLLECTION_SYNC,
        JOB_CREATE_COLLECTIONS,
        JOB_DRY_RUN,
        JOB_HEALTHCHECK,
        JOB_MEDIA_UPLOAD,
        JOB_PARSE_CSV,
        JOB_PRODUCT_SYNC,
        JOB_RUN_FLOW,
        load_wix_utility_env,
    )
    from wix_utility.jobs.dispatcher import run_job
else:
    from wix_utility.core.config import (
        DEFAULT_JOB_NAME,
        ENV_DEFAULT_JOB,
        JOB_COLLECTION_SYNC,
        JOB_CREATE_COLLECTIONS,
        JOB_DRY_RUN,
        JOB_HEALTHCHECK,
        JOB_MEDIA_UPLOAD,
        JOB_PARSE_CSV,
        JOB_PRODUCT_SYNC,
        JOB_RUN_FLOW,
        load_wix_utility_env,
    )
    from wix_utility.jobs.dispatcher import run_job


def main(argv: list[str] | None = None) -> int:
    load_wix_utility_env()

    parser = argparse.ArgumentParser(description="Run a Wix utility job.")
    parser.add_argument(
        "job",
        nargs="?",
        help=(
            "Job to run "
            f"({JOB_HEALTHCHECK}, {JOB_DRY_RUN}, {JOB_PARSE_CSV}, {JOB_RUN_FLOW}, {JOB_CREATE_COLLECTIONS}, "
            f"{JOB_COLLECTION_SYNC}, {JOB_PRODUCT_SYNC}, {JOB_MEDIA_UPLOAD}). "
            f"If omitted, reads {ENV_DEFAULT_JOB} from wix_utility/.env "
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
