"""Parse CSV job."""

from __future__ import annotations

from wix_utility.core.config import JOB_PARSE_CSV, WixConfig
from wix_utility.flows.csv_parse import ParseCsvFlow
from wix_utility.jobs.base import WixJob


class ParseCsvJob(WixJob):
    job_name = JOB_PARSE_CSV

    def run(self, config: WixConfig) -> int:
        return ParseCsvFlow().execute(config).exit_code
