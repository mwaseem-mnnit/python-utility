"""Healthcheck job."""

from __future__ import annotations

from wix_utility.core.config import JOB_HEALTHCHECK, WixConfig
from wix_utility.jobs.base import WixJob


class HealthcheckJob(WixJob):
    job_name = JOB_HEALTHCHECK

    def run(self, config: WixConfig) -> int:
        self.logger.info("Wix utility healthcheck")
        self.logger.info("Base URL: %s", config.base_url)
        self.logger.info("Site ID configured: %s", bool(config.site_id))
        self.logger.info("API key configured: %s", bool(config.api_key))
        self.logger.info("Dry run: %s", config.dry_run)
        return 0
