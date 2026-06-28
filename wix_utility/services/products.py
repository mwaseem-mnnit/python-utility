"""Product operations for Wix Stores catalog workflows."""

from __future__ import annotations

from typing import Any

from wix_utility.catalog.models import ProductDraft
from wix_utility.clients.wix_api import WixApiClient


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
