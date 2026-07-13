"""Job registry for Wix utility commands."""

from __future__ import annotations

import sys

from wix_utility.core.config import (
    JOB_ASSIGN_COLLECTION_TO_PRODUCT,
    JOB_COLLECTION_PRODUCTS_TO_CMS,
    JOB_COLLECTION_SYNC,
    JOB_CREATE_COLLECTIONS,
    JOB_DRY_RUN,
    JOB_HEALTHCHECK,
    JOB_MEDIA_UPLOAD,
    JOB_PARSE_CSV,
    JOB_PRODUCT_SYNC,
    JOB_RUN_FLOW,
)
from wix_utility.jobs.base import WixJob
from wix_utility.jobs.assign_collection_to_product import AssignCollectionToProductJob
from wix_utility.jobs.collection_products_to_cms import CollectionProductsToCmsJob
from wix_utility.jobs.create_collections import CreateCollectionsJob
from wix_utility.jobs.dry_run import DryRunJob
from wix_utility.jobs.flow_jobs import CollectionSyncJob, MediaUploadJob, ProductSyncJob
from wix_utility.jobs.healthcheck import HealthcheckJob
from wix_utility.jobs.parse_csv import ParseCsvJob
from wix_utility.jobs.run_flow import RunFlowJob


def job_registry() -> dict[str, type[WixJob]]:
    return {
        JOB_HEALTHCHECK: HealthcheckJob,
        JOB_DRY_RUN: DryRunJob,
        JOB_CREATE_COLLECTIONS: CreateCollectionsJob,
        JOB_ASSIGN_COLLECTION_TO_PRODUCT: AssignCollectionToProductJob,
        JOB_COLLECTION_PRODUCTS_TO_CMS: CollectionProductsToCmsJob,
        JOB_PARSE_CSV: ParseCsvJob,
        JOB_RUN_FLOW: RunFlowJob,
        JOB_COLLECTION_SYNC: CollectionSyncJob,
        JOB_PRODUCT_SYNC: ProductSyncJob,
        JOB_MEDIA_UPLOAD: MediaUploadJob,
    }


def run_job(job: str) -> int:
    key = job.strip().lower()
    jobs = job_registry()
    job_cls = jobs.get(key)
    if job_cls is None:
        valid = ", ".join(sorted(jobs))
        print(f"Unknown WIX job '{job}'. Valid jobs: {valid}", file=sys.stderr)
        return 2
    return job_cls().execute()
