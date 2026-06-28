"""HTTP client wrapper for Wix REST API calls."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests import Response
from requests.exceptions import RequestException

from wix_utility.core.config import WixConfig
from wix_utility.core.utils import payload_fingerprint

logger = logging.getLogger(__name__)


class WixApiError(RuntimeError):
    """Raised when Wix returns a non-success response after retries."""


class WixApiClient:
    """Thin retrying JSON client with dry-run support."""

    def __init__(self, config: WixConfig) -> None:
        self.config = config

    def headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = self.config.api_key
        if self.config.cookie:
            headers["Cookie"] = self.config.cookie
        if self.config.site_id:
            headers["wix-site-id"] = self.config.site_id
        if self.config.account_id:
            headers["wix-account-id"] = self.config.account_id
        return headers

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

        if not self.config.has_credentials:
            raise WixApiError("WIX_API_KEY is required when WIX_DRY_RUN=false")

        for attempt in range(1, self.config.max_retries + 2):
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=self.headers(),
                    json=json_payload,
                    params=params,
                    timeout=self.config.request_timeout_seconds,
                )
                if 200 <= response.status_code < 300:
                    logger.info("%s %s ok status=%s", method.upper(), path, response.status_code)
                    return self._decode_json(response)

                logger.warning(
                    "%s %s failed attempt=%s status=%s body=%s",
                    method.upper(),
                    path,
                    attempt,
                    response.status_code,
                    response.text[:500],
                )
            except RequestException as exc:
                logger.warning("%s %s error attempt=%s error=%s", method.upper(), path, attempt, exc)

            if attempt <= self.config.max_retries:
                time.sleep(self.config.retry_backoff_seconds * attempt)

        raise WixApiError(f"{method.upper()} {path} failed after retries")

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("GET", path, params=params)

    def post(self, path: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", path, json_payload=json_payload)

    def patch(self, path: str, *, json_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("PATCH", path, json_payload=json_payload)

    def delete(self, path: str) -> dict[str, Any]:
        return self.request("DELETE", path)

    def _url(self, path: str) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self.config.base_url}{clean_path}"

    @staticmethod
    def _decode_json(response: Response) -> dict[str, Any]:
        if not response.content:
            return {}
        try:
            decoded = response.json()
        except ValueError as exc:
            raise WixApiError(f"Response was not JSON: {response.text[:200]}") from exc
        if not isinstance(decoded, dict):
            return {"data": decoded}
        return decoded
