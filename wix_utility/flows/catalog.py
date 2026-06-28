"""Top-level catalog sync flow boilerplate."""

from __future__ import annotations

from wix_utility.core.config import FLOW_CATALOG_SYNC, WixConfig
from wix_utility.core.utils import ensure_output_dir
from wix_utility.flows.base import FlowResult, WixFlow
from wix_utility.flows.helpers import load_optional_csv


class CatalogSyncFlow(WixFlow):
    flow_name = FLOW_CATALOG_SYNC

    def run(self, config: WixConfig) -> FlowResult:
        ensure_output_dir(config.output_dir)
        records = load_optional_csv(config)
        self.logger.info("Catalog sync boilerplate loaded records=%s", len(records))
        self.logger.info("Next step will orchestrate collection sync, product sync, and media upload.")
        return FlowResult(exit_code=0, flow_name=self.flow_name, records_loaded=len(records))
