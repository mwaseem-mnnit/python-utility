"""Export collection-scoped Wix products into a CMS collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wix_utility.clients.wix_api import WixApiClient, WixApiError
from wix_utility.core.config import JOB_COLLECTION_PRODUCTS_TO_CMS, WixConfig
from wix_utility.core.utils import (
    chunked,
    compute_next_offset,
    convert_to_wix_media_url,
    random_decimal_between,
    random_number_between,
)
from wix_utility.jobs.base import WixJob
from wix_utility.services.cms import CmsService
from wix_utility.services.products import ProductService


@dataclass(frozen=True)
class CmsExportStats:
    collection_id: str
    fetched_products: int
    exported_items: int


class CollectionProductsToCmsJob(WixJob):
    """Read Wix store products for selected collections and upsert CMS rows."""

    job_name = JOB_COLLECTION_PRODUCTS_TO_CMS

    def run(self, config: WixConfig) -> int:
        if not config.filter_collection_id_list:
            self.logger.error("No collection ids provided. Set filterCollectionIdList in the env file.")
            return 1
        if not config.target_cms_table_id:
            self.logger.error("No CMS table id provided. Set targetCMSTableId in the env file.")
            return 1
        if config.batch_size <= 0:
            self.logger.error("batchSize must be greater than zero.")
            return 1

        api_client = WixApiClient(config)
        product_service = ProductService(api_client)
        cms_service = CmsService(api_client)

        self.logger.info(
            "Starting CMS export for %s collection ids into table=%s",
            len(config.filter_collection_id_list),
            config.target_cms_table_id,
        )

        exported: dict[str, dict[str, Any]] = {}
        stats: list[CmsExportStats] = []
        for collection_id in config.filter_collection_id_list:
            try:
                products = self._load_collection_products(
                    product_service,
                    collection_id=collection_id,
                    page_size=max(1, config.product_page_size),
                )
            except WixApiError as exc:
                self.logger.error("Failed to query products for collection %s: %s", collection_id, exc)
                return 1

            self.logger.info("Collection %s: fetched %s products", collection_id, len(products))
            exported_count = 0
            for product in products:
                record = self._build_cms_record(product, config)
                if record is None:
                    continue
                key = record["slug"]
                if key in exported:
                    continue
                exported[key] = record
                exported_count += 1

            stats.append(
                CmsExportStats(
                    collection_id=collection_id,
                    fetched_products=len(products),
                    exported_items=exported_count,
                )
            )

        self.logger.info("Prepared %s CMS rows", len(exported))
        for stat in stats:
            self.logger.info(
                "Collection %s summary: fetched=%s exported=%s",
                stat.collection_id,
                stat.fetched_products,
                stat.exported_items,
            )

        records = list(exported.values())
        batches = chunked(records, config.batch_size)
        self.logger.info("Upserting %s batches with batchSize=%s", len(batches), config.batch_size)

        failures = 0
        for index, batch in enumerate(batches, start=1):
            try:
                self.logger.info("Upserting batch %s/%s size=%s", index, len(batches), len(batch))
                cms_service.bulk_upsert_cms(config.target_cms_table_id, batch)
            except WixApiError as exc:
                failures += 1
                self.logger.error("Failed batch %s/%s: %s", index, len(batches), exc)

        self.logger.info(
            "CMS export complete: exported=%s batches=%s failures=%s",
            len(records),
            len(batches),
            failures,
        )
        return 1 if failures else 0

    def _load_collection_products(
        self,
        product_service: ProductService,
        *,
        collection_id: str,
        page_size: int,
    ) -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = product_service.query_products_page(
                limit=page_size,
                offset=offset,
                collection_ids=collection_id,
            )
            page_items = _extract_products(page)
            if not page_items:
                break
            products.extend(page_items)
            next_offset = compute_next_offset(page)
            if next_offset <= offset:
                next_offset = offset + len(page_items)
            if next_offset <= offset:
                break
            offset = next_offset
        return products

    def _build_cms_record(self, product: dict[str, Any], config: WixConfig) -> dict[str, Any] | None:
        slug = str(_get_product_value(product, "slug") or "").strip()
        title = str(_get_product_value(product, "name", "title") or "").strip()
        if not slug or not title:
            self.logger.warning("Skipping product with missing slug/title: %s", _safe_product_debug(product))
            return None

        media_url = _get_nested_value(product, ("media", "mainMedia", "thumbnail", "url"))
        if not media_url:
            media_url = _get_nested_value(product, ("mainMedia", "thumbnail", "url"))
        if not media_url:
            self.logger.warning("Skipping product %s because media thumbnail url is missing", slug)
            return None

        price_data = _get_nested_value(product, ("priceData",))
        if not isinstance(price_data, dict):
            self.logger.warning("Skipping product %s because priceData is missing", slug)
            return None

        discount_value = _get_nested_value(product, ("discount", "value"))
        final_price = price_data.get("discountedPrice")
        original_price = price_data.get("price")
        if final_price is None or original_price is None:
            self.logger.warning("Skipping product %s because price values are missing", slug)
            return None

        rating = random_decimal_between(
            config.min_rating,
            config.max_rating,
            precision=config.precision,
        )
        review_count = random_number_between(config.review_count_from, config.review_count_to)

        try:
            main_media = convert_to_wix_media_url(
                str(media_url),
                media_width=config.media_width,
                media_height=config.media_height,
            )
        except ValueError as exc:
            self.logger.warning("Skipping product %s because media url could not be converted: %s", slug, exc)
            return None

        return {
            "slug": slug,
            "title": title,
            "discountPercent": float(discount_value or 0),
            "version": config.cms_record_version,
            "rating": rating,
            "finalPrice": float(final_price),
            "originalPrice": float(original_price),
            "reviewCount": float(review_count),
            "mainMedia": main_media,
        }


def _extract_products(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        payload.get("products"),
        payload.get("items"),
        payload.get("data", {}).get("products") if isinstance(payload.get("data"), dict) else None,
        payload.get("data", {}).get("items") if isinstance(payload.get("data"), dict) else None,
    )
    for value in candidates:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _get_product_value(product: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _get_nested_value(product, (path,))
        if value is not None:
            return value
    return None


def _get_nested_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _safe_product_debug(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "slug": product.get("slug"),
        "name": product.get("name"),
        "title": product.get("title"),
    }
