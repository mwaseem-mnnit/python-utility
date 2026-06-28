"""Flow registry."""

from __future__ import annotations

import logging

from wix_utility.core.config import (
    FLOW_CATALOG_SYNC,
    FLOW_COLLECTION_SYNC,
    FLOW_MEDIA_UPLOAD,
    FLOW_PRODUCT_SYNC,
    WixConfig,
)
from wix_utility.flows.base import FlowResult, WixFlow
from wix_utility.flows.catalog import CatalogSyncFlow
from wix_utility.flows.collections import CollectionSyncFlow
from wix_utility.flows.media import MediaUploadFlow
from wix_utility.flows.products import ProductSyncFlow

logger = logging.getLogger(__name__)


def flow_registry() -> dict[str, type[WixFlow]]:
    return {
        FLOW_CATALOG_SYNC: CatalogSyncFlow,
        FLOW_COLLECTION_SYNC: CollectionSyncFlow,
        FLOW_PRODUCT_SYNC: ProductSyncFlow,
        FLOW_MEDIA_UPLOAD: MediaUploadFlow,
    }


def run_flow(flow_name: str, config: WixConfig) -> FlowResult:
    flows = flow_registry()
    key = flow_name.strip().lower()
    flow_cls = flows.get(key)
    if flow_cls is None:
        valid = ", ".join(sorted(flows))
        logger.error("Unknown WIX_FLOW '%s'. Valid flows: %s", flow_name, valid)
        return FlowResult(exit_code=2, flow_name=flow_name, message=f"valid flows: {valid}")
    return flow_cls().execute(config)
