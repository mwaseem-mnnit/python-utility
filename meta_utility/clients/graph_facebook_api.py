"""HTTP client wrapper for Graph Facebook API calls."""

from __future__ import annotations

import logging
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException

from meta_utility.core.config import MetaConfig
from meta_utility.core.utils import payload_fingerprint

logger = logging.getLogger(__name__)


class GraphFacebookApiError(RuntimeError):
    """Raised when Graph Facebook API returns a non-success response."""


class GraphFacebookApiClient:
    """Thin JSON client for Graph Facebook API."""

    def __init__(self, config: MetaConfig) -> None:
        self.config = config

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.graph_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a JSON request and return the decoded response body."""
        url = self._url(path)
        fingerprint = payload_fingerprint(json_payload or params or {})

        if self.config.dry_run:
            logger.info("Dry run %s %s payload=%s", method.upper(), url, fingerprint[:12])
            return {
                "dryRun": True,
                "method": method.upper(),
                "url": url,
                "payloadFingerprint": fingerprint,
            }
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self.headers(),
                json=json_payload,
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
        except RequestException as exc:
            logger.error("%s %s request error: %s", method.upper(), path, exc)
            raise GraphFacebookApiError(f"{method.upper()} {path} request failed") from exc

        if not 200 <= response.status_code < 300:
            logger.error(
                "%s %s failed status=%s body=%s",
                method.upper(),
                path,
                response.status_code,
                response.text[:500],
            )
            raise GraphFacebookApiError(
                f"{method.upper()} {path} failed with status {response.status_code}"
            )

        logger.info("%s %s ok status=%s", method.upper(), path, response.status_code)
        return self._decode_json(response)

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(self, path: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", path, json_payload=json_payload)

    def patch(self, path: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("PATCH", path, json_payload=json_payload)

    def delete(self, path: str) -> dict[str, Any]:
        return self.request("DELETE", path)

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self.config.graph_base_url}{clean_path}"

    @staticmethod
    def _decode_json(response: Response) -> dict[str, Any]:
        if not response.content:
            return {}
        try:
            decoded = response.json()
        except ValueError as exc:
            raise GraphFacebookApiError(f"Response was not JSON: {response.text[:200]}") from exc
        if isinstance(decoded, dict):
            return decoded
        return {"data": decoded}

