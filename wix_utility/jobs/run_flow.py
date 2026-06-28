"""Run the configured flow job."""

from __future__ import annotations

from wix_utility.core.config import JOB_RUN_FLOW, WixConfig
from wix_utility.flows.registry import run_flow
from wix_utility.jobs.base import WixJob


class RunFlowJob(WixJob):
    job_name = JOB_RUN_FLOW

    def run(self, config: WixConfig) -> int:
        return run_flow(config.flow_name, config).exit_code
