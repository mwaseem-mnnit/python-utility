"""Collection/category operations for Wix Stores catalog workflows."""

from __future__ import annotations

from typing import Any

from wix_utility.catalog.models import CollectionDraft
from wix_utility.clients.wix_api import WixApiClient
from wix_utility.core.utils import _compute_next_offset


def collection_payload(draft: CollectionDraft) -> dict[str, Any]:
    """Build a neutral collection payload; endpoint-specific mapping can evolve here."""
    payload: dict[str, Any] = {
        "name": draft.name,
    }
    if draft.slug:
        payload["slug"] = draft.slug
    if draft.description:
        payload["description"] = draft.description
    if draft.visible is not None:
        payload["visible"] = draft.visible
    if draft.metadata:
        payload["metadata"] = draft.metadata
    return payload


class CollectionService:
    def __init__(self, client: WixApiClient) -> None:
        self.client = client

    def create_collection(self, draft: CollectionDraft) -> dict[str, Any]:
        """Create a Wix collection/category."""
        return self.client.post(
            "/stores/v1/collections",
            json_payload={"collection": collection_payload(draft)},
        )

    def query_collections_page(
        self,
        *,
        limit: int,
        offset: int = 0,
        visible_only: bool = True,
    ) -> dict[str, Any]:
        """Query one page of Wix Stores collections."""
        paging: dict[str, Any] = {"limit": limit, "offset": offset}
        query: dict[str, Any] = {"paging": paging}
        if visible_only:
            query["filter"] = "{\"visible\": true}"

        return self.client.post(
            "/stores/v1/collections/query",
            json_payload={"query": query},
        )

    def list_collections(self, *, page_size: int, visible_only: bool = True) -> list[dict[str, Any]]:
        """Read all collection pages from Wix Stores Catalog v1."""
        collections: list[dict[str, Any]] = []
        offset: int = 0
        while True:
            page = self.query_collections_page(
                limit=page_size,
                offset=offset,
                visible_only=visible_only
            )
            page_items = _extract_collections(page)
            if len(page_items) == 0:
                break
            collections.extend(page_items)
            offset = _compute_next_offset(page)
        return collections


def _extract_collections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = (
        payload.get("collections"),
        payload.get("items"),
        payload.get("data", {}).get("collections") if isinstance(payload.get("data"), dict) else None,
        payload.get("data", {}).get("items") if isinstance(payload.get("data"), dict) else None,
    )
    for value in candidates:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_next_cursor(payload: dict[str, Any]) -> str:
    metadata_candidates = [
        payload.get("metadata"),
        payload.get("pagingMetadata"),
        payload.get("data", {}).get("metadata") if isinstance(payload.get("data"), dict) else None,
        payload.get("data", {}).get("pagingMetadata") if isinstance(payload.get("data"), dict) else None,
    ]
    for metadata in metadata_candidates:
        if not isinstance(metadata, dict):
            continue
        cursors = metadata.get("cursors")
        if isinstance(cursors, dict):
            next_cursor = str(cursors.get("next") or "").strip()
            if next_cursor:
                return next_cursor
        next_cursor = str(metadata.get("nextCursor") or "").strip()
        if next_cursor:
            return next_cursor
    return ""
