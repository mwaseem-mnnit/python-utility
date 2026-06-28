"""Direct job wrappers around individual flows."""

from __future__ import annotations

from wix_utility.core.config import (
    JOB_COLLECTION_SYNC,
    JOB_MEDIA_UPLOAD,
    JOB_PRODUCT_SYNC,
    WixConfig,
)
from wix_utility.flows.collections import CollectionSyncFlow
from wix_utility.flows.media import MediaUploadFlow
from wix_utility.flows.products import ProductSyncFlow
from wix_utility.jobs.base import WixJob


class CollectionSyncJob(WixJob):
    job_name = JOB_COLLECTION_SYNC

    def run(self, config: WixConfig) -> int:
        return CollectionSyncFlow().execute(config).exit_code


class ProductSyncJob(WixJob):
    job_name = JOB_PRODUCT_SYNC

    def run(self, config: WixConfig) -> int:
        return ProductSyncFlow().execute(config).exit_code


class MediaUploadJob(WixJob):
    job_name = JOB_MEDIA_UPLOAD

    def run(self, config: WixConfig) -> int:
        return MediaUploadFlow().execute(config).exit_code
