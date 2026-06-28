"""Collection sync flow boilerplate."""

from __future__ import annotations

from wix_utility.core.config import FLOW_COLLECTION_SYNC, WixConfig
from wix_utility.flows.base import FlowResult, WixFlow
from wix_utility.flows.helpers import load_optional_csv


class CollectionSyncFlow(WixFlow):
    flow_name = FLOW_COLLECTION_SYNC

    def run(self, config: WixConfig) -> FlowResult:
        records = load_optional_csv(config)
        self.logger.info("Collection sync boilerplate loaded records=%s", len(records))
        self.logger.info("Next step will compare existing Wix collections and create/update only missing changes.")
        return FlowResult(exit_code=0, flow_name=self.flow_name, records_loaded=len(records))
