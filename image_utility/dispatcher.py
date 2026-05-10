"""Job registry for image utilities."""

from __future__ import annotations

import sys
from collections.abc import Callable

from .config import JOB_COMPRESS, JOB_ENHANCE, JOB_PIPELINE

JobRunner = Callable[[], int]


def _registry() -> dict[str, JobRunner]:
    from image_utility import compress, legacy_enhance_img
    from image_utility.pipeline.runner import run as pipeline_run

    return {
        JOB_COMPRESS: compress.run,
        JOB_PIPELINE: pipeline_run,
        JOB_ENHANCE: legacy_enhance_img.run
    }


def run_job(job: str) -> int:
    from image_utility.utils import load_image_utility_env

    load_image_utility_env()
    key = job.strip().lower()
    jobs = _registry()
    runner = jobs.get(key)
    if runner is None:
        valid = ", ".join(sorted(jobs))
        print(f"Unknown IMAGE_JOB '{job}'. Valid jobs: {valid}", file=sys.stderr)
        return 2
    return runner()
