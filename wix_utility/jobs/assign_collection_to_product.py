"""Assign collections to products using CSV mapping and Wix APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wix_utility.clients.wix_api import WixApiClient, WixApiError
from wix_utility.core.config import JOB_ASSIGN_COLLECTION_TO_PRODUCT, WixConfig
from wix_utility.core.utils import dict_from_list_by_key
from wix_utility.io.csv_records import parse_csv_records
from wix_utility.jobs.base import WixJob
from wix_utility.services.collections import CollectionService
from wix_utility.services.products import ProductService


@dataclass(frozen=True)
class AssignmentMapping:
    """Mapping of collection ID to list of product IDs for bulk assignment."""

    collection_id: str
    product_ids: list[str]


class AssignCollectionToProductJob(WixJob):
    """Assign collections to products from a CSV mapping."""

    job_name = JOB_ASSIGN_COLLECTION_TO_PRODUCT

    def run(self, config: WixConfig) -> int:
        # Load CSV records
        csv_records = self._load_csv_records(config)
        if csv_records is None:
            return 1

        self.logger.info("Loaded CSV records: %s", len(csv_records))

        # Initialize API client and services
        api_client = WixApiClient(config)
        collection_service = CollectionService(api_client)
        product_service = ProductService(api_client)

        # Fetch all collections
        try:
            self.logger.info("Fetching collections with page_size=%s", config.collection_page_size)
            collections = collection_service.list_collections(page_size=max(1, config.collection_page_size))
            self.logger.info("Loaded collections: %s", len(collections))
        except WixApiError as exc:
            self.logger.error("Failed to fetch collections: %s", exc)
            return 1

        # Fetch all products
        try:
            self.logger.info("Fetching products with page_size=%s", config.product_page_size)
            products = product_service.list_products(page_size=max(1, config.product_page_size))
            self.logger.info("Loaded products: %s", len(products))
        except WixApiError as exc:
            self.logger.error("Failed to fetch products: %s", exc)
            return 1

        # Create slug-based lookup dictionaries
        collection_by_slug = dict_from_list_by_key(collections, "slug")
        product_by_slug = dict_from_list_by_key(products, "slug")

        self.logger.info(
            "Collection slug mapping: %s unique slugs | Product slug mapping: %s unique slugs",
            len(collection_by_slug),
            len(product_by_slug),
        )

        # Build assignment mappings
        assignment_mappings = self._build_assignment_mappings(
            csv_records,
            product_by_slug,
            collection_by_slug,
        )

        if not assignment_mappings:
            self.logger.warning("No valid assignment mappings found from CSV")
            return 0

        self.logger.info("Built %s assignment mappings", len(assignment_mappings))

        # Apply assignments
        return self._apply_assignments(api_client, assignment_mappings, config.dry_run)

    def _load_csv_records(self, config: WixConfig) -> list[dict[str, Any]] | None:
        """Load CSV records from WIX_INPUT_CSV."""
        if config.input_csv is None:
            self.logger.error("WIX_INPUT_CSV is not set")
            return None

        if not config.input_csv.is_file():
            self.logger.error("CSV file does not exist: %s", config.input_csv)
            return None

        try:
            records = parse_csv_records(config.input_csv, delimiter=config.csv_delimiter)
            return records
        except Exception as exc:
            self.logger.error("Failed to parse CSV: %s", exc)
            return None

    def _build_assignment_mappings(
        self,
        csv_records: list[dict[str, Any]],
        product_by_slug: dict[str, dict[str, Any]],
        collection_by_slug: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Build mapping from collection ID to list of product IDs.
        
        Returns a dict mapping collection_id → [product_id, ...].
        """
        assignments: dict[str, list[str]] = {}
        skipped = 0
        failed_matches = 0

        for index, record in enumerate(csv_records, start=2):
            product_slug = str(record.get("product_slug") or "").strip()
            collection_slug = str(record.get("collection_slug") or "").strip()

            if not product_slug or not collection_slug:
                self.logger.warning(
                    "Row %s: skipping - missing product_slug or collection_slug",
                    index,
                )
                skipped += 1
                continue

            # Look up product by slug (handleId maps to product slug)
            product = product_by_slug.get(product_slug)
            if not product:
                self.logger.warning(
                    "Row %s: skipping - product with slug %r not found",
                    index,
                    product_slug,
                )
                failed_matches += 1
                continue

            # Look up collection by slug
            collection = collection_by_slug.get(collection_slug)
            if not collection:
                self.logger.warning(
                    "Row %s: skipping - collection with slug %r not found",
                    index,
                    collection_slug,
                )
                failed_matches += 1
                continue

            product_id = product.get("id")
            collection_id = collection.get("id")

            if not product_id or not collection_id:
                self.logger.warning(
                    "Row %s: skipping - missing product id or collection id",
                    index,
                )
                failed_matches += 1
                continue

            # Add to assignments
            if collection_id not in assignments:
                assignments[collection_id] = []
            assignments[collection_id].append(product_id)
            self.logger.debug(
                "Row %s: mapped product %r to collection %r",
                index,
                product_id,
                collection_id,
            )

        self.logger.info(
            "CSV processing complete: matched=%s skipped=%s failed_matches=%s",
            sum(len(v) for v in assignments.values()),
            skipped,
            failed_matches,
        )
        return assignments

    def _apply_assignments(
        self,
        api_client: WixApiClient,
        assignments: dict[str, list[str]],
        dry_run: bool,
    ) -> int:
        """Apply collection assignments via Wix API."""
        total_assigned = 0
        failed = 0

        for collection_id, product_ids in assignments.items():
            self.logger.info(
                "Assigning %s products to collection %s",
                len(product_ids),
                collection_id,
            )

            try:
                if dry_run:
                    self.logger.info(
                        "Dry-run: would assign products %s to collection %s",
                        product_ids,
                        collection_id,
                    )
                    total_assigned += len(product_ids)
                else:
                    api_client.post(
                        f"/stores/v1/collections/{collection_id}/productIds",
                        json_payload={"productIds": product_ids},
                    )
                    total_assigned += len(product_ids)
                    self.logger.info(
                        "Successfully assigned %s products to collection %s",
                        len(product_ids),
                        collection_id,
                    )
            except WixApiError as exc:
                failed += 1
                self.logger.error(
                    "Failed to assign products to collection %s: %s",
                    collection_id,
                    exc,
                )

        self.logger.info(
            "Assignment complete: total_assigned=%s failed_collections=%s",
            total_assigned,
            failed,
        )
        return 1 if failed else 0
