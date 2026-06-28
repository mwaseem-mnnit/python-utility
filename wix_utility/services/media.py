"""Media upload placeholders for Wix product images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wix_utility.clients.wix_api import WixApiClient


class MediaService:
    def __init__(self, client: WixApiClient) -> None:
        self.client = client

    def prepare_upload(self, path: Path, *, mime_type: str = "image/jpeg") -> dict[str, Any]:
        """Request an upload target for a local image file.

        Actual binary upload support will be added once the preferred Wix media
        flow is selected for the store.
        """
        return self.client.post(
            "/site-media/v1/files/generate-upload-url",
            json_payload={
                "fileName": path.name,
                "mimeType": mime_type,
            },
        )
