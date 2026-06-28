"""Dry-run config inspection job."""

from __future__ import annotations

from wix_utility.core.config import JOB_DRY_RUN, WixConfig
from wix_utility.jobs.base import WixJob


class DryRunJob(WixJob):
    job_name = JOB_DRY_RUN

    def run(self, config: WixConfig) -> int:
        self.logger.info("Dry-run config loaded for input_csv=%s image_dir=%s", config.input_csv, config.image_dir)
        return 0
