"""Base classes for Wix flow runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from wix_utility.core.config import WixConfig


@dataclass(frozen=True)
class FlowResult:
    """Summary returned by a flow runner."""

    exit_code: int
    flow_name: str
    records_loaded: int = 0
    message: str = ""


class WixFlow(ABC):
    """Base class for no-mutation flow boilerplate."""

    flow_name: str

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__module__)

    def execute(self, config: WixConfig) -> FlowResult:
        self.logger.info("Starting flow: %s", self.flow_name)
        result = self.run(config)
        self.logger.info("Finished flow: %s exit_code=%s", self.flow_name, result.exit_code)
        return result

    @abstractmethod
    def run(self, config: WixConfig) -> FlowResult:
        """Run the flow and return a summary."""
