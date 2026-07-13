"""Product operations for Wix Stores catalog workflows."""

from __future__ import annotations

import json
from typing import Any

from wix_utility.catalog.models import ProductDraft
from wix_utility.clients.wix_api import WixApiClient
from wix_utility.core.utils import compute_next_offset


def product_payload(draft: ProductDraft) -> dict[str, Any]:
    """Build a neutral product payload from locally parsed product data."""
    payload: dict[str, Any] = {
        "name": draft.name,
    }
    if draft.sku:
        payload["sku"] = draft.sku
    if draft.description:
        payload["description"] = draft.description
    if draft.price is not None:
        payload["price"] = draft.price
    if draft.collections:
        payload["collections"] = list(draft.collections)
    if draft.images:
        payload["images"] = [
            {
                "path": str(image.path),
                "altText": image.alt_text,
                "wixMediaId": image.wix_media_id,
            }
            for image in draft.images
        ]
    if draft.metadata:
        payload["metadata"] = draft.metadata
    return payload


class ProductService:
    def __init__(self, client: WixApiClient) -> None:
        self.client = client

    def create_product(self, draft: ProductDraft) -> dict[str, Any]:
        """Create a Wix product draft."""
        return self.client.post(
            "/stores/v1/products",
            json_payload={"product": product_payload(draft)},
        )

    def query_products_page(
        self,
        *,
        limit: int,
        offset: int = 0,
        collection_ids: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Query one page of Wix Stores products."""
        paging: dict[str, Any] = {"limit": limit, "offset": offset}
        query: dict[str, Any] = {"paging": paging}
        filter_payload = _build_product_filter(collection_ids)
        if filter_payload is not None:
            query["filter"] = json.dumps(filter_payload)

        return self.client.post(
            "/stores/v1/products/query",
            json_payload={"query": query},
        )

    def list_products(
        self,
        *,
        page_size: int,
        collection_ids: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Read all product pages from Wix Stores Catalog v1."""
        products: list[dict[str, Any]] = []
        offset: int = 0
        while True:
            page = self.query_products_page(
                limit=page_size,
                offset=offset,
                collection_ids=collection_ids,
            )
            page_items = _extract_products(page)
            if len(page_items) == 0:
                break
            products.extend(page_items)
            next_offset = compute_next_offset(page)
            if next_offset <= offset:
                next_offset = offset + len(page_items)
            if next_offset <= offset:
                break
            offset = next_offset

        return products


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


def _build_product_filter(collection_ids: str | list[str] | None) -> dict[str, Any] | None:
    if collection_ids is None:
        return None
    if isinstance(collection_ids, str):
        collection_id = collection_ids.strip()
        if not collection_id:
            return None
        return {"collectionIds": collection_id}

    values = [collection_id.strip() for collection_id in collection_ids if collection_id.strip()]
    if not values:
        return None
    if len(values) == 1:
        return {"collectionIds": values[0]}
    return {"collectionIds": values}