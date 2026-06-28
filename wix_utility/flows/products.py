"""Product sync flow boilerplate."""

from __future__ import annotations

from wix_utility.core.config import FLOW_PRODUCT_SYNC, WixConfig
from wix_utility.flows.base import FlowResult, WixFlow
from wix_utility.flows.helpers import load_optional_csv


class ProductSyncFlow(WixFlow):
    flow_name = FLOW_PRODUCT_SYNC

    def run(self, config: WixConfig) -> FlowResult:
        records = load_optional_csv(config)
        self.logger.info("Product sync boilerplate loaded records=%s", len(records))
        self.logger.info("Next step will map rows to ProductDraft objects and create/update Wix products.")
        return FlowResult(exit_code=0, flow_name=self.flow_name, records_loaded=len(records))
