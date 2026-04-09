from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_HEADERS = {
    "User-Agent": _DEFAULT_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_ATTEMPTS = 2


def fetch_html(url: str, *, timeout: float = 30.0) -> str:
    """
    GET ``url`` and return response text.

    Retries once on failure (2 attempts total). Raises the last exception if both fail.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except (requests.RequestException, OSError) as e:
            last_exc = e
            logger.warning("fetch_html attempt %s/%s failed for %s: %s", attempt, _ATTEMPTS, url, e)
    assert last_exc is not None
    raise last_exc
