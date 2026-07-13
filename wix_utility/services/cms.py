"""Wix CMS bulk save helpers."""

from __future__ import annotations

from typing import Any

from wix_utility.clients.wix_api import WixApiClient


class CmsService:
    def __init__(self, client: WixApiClient) -> None:
        self.client = client

    def bulk_upsert_cms(
        self,
        collection_id: str,
        data_objects: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return bulkUpsertCMS(self.client, collection_id, data_objects)


def bulkUpsertCMS(
    client: WixApiClient,
    collection_id: str,
    data_objects: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save a batch of CMS items into a Wix data collection."""
    data_items: list[dict[str, Any]] = []
    for item in data_objects:
        if not isinstance(item, dict):
            raise TypeError("data_objects must contain dictionaries")
        data_items.append({"data": item})

    return client.post(
        "/wix-data/v2/bulk/items/save",
        json_payload={
            "dataCollectionId": collection_id,
            "dataItems": data_items,
        },
    )


bulk_upsert_cms = bulkUpsertCMS
