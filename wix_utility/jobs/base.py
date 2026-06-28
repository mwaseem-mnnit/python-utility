"""Base class for Wix CLI jobs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from wix_utility.core.config import WixConfig, load_wix_config
from wix_utility.core.logging import init_job_logging


class WixJob(ABC):
    """Base job with shared logging and config loading."""

    job_name: str
    log_filename = "wix_utility.log"

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__module__)

    def execute(self) -> int:
        init_job_logging(self.log_filename)
        self.logger = logging.getLogger(self.__class__.__module__)
        config = load_wix_config()
        self.logger.info("Starting job: %s", self.job_name)
        exit_code = self.run(config)
        self.logger.info("Finished job: %s exit_code=%s", self.job_name, exit_code)
        return exit_code

    @abstractmethod
    def run(self, config: WixConfig) -> int:
        """Run the job and return a process exit code."""
