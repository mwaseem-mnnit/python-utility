"""
Fetch HTML over HTTP and parse it for downstream scraping tasks.

Returns a :class:`ParsedHtmlPage` so callers can use either the
:class:`~bs4.BeautifulSoup` tree (navigation, CSS selectors) or the raw
``raw_html`` string (regex, exact snippets, saving to disk).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

import scrape_web.site_2.config  # noqa: F401 — env + logging

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class ParsedHtmlPage:
    """
    Result of fetching and parsing a document.

    * ``soup`` — use for CSS selectors, ``find`` / ``find_all``, structure.
    * ``raw_html`` — decoded response body; use when you need the exact string
      (search, audit, pass to another tool).
    """

    url: str
    final_url: str
    status_code: int
    soup: BeautifulSoup
    raw_html: str
    encoding: str | None


def fetch_and_parse_html(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    session: requests.Session | None = None,
) -> ParsedHtmlPage:
    """
    GET ``url``, decode the body, and parse HTML with BeautifulSoup (``html.parser``).

    Raises :class:`requests.HTTPError` on 4xx/5xx and :class:`requests.RequestException`
    on network errors.
    """
    if not url or not str(url).strip():
        raise ValueError("url must be non-empty")

    target = str(url).strip()
    logger.info("GET %s", target)

    client = session or requests.Session()
    try:
        resp = client.get(
            target,
            headers=_REQUEST_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("request failed: %s", target)
        raise

    encoding = resp.encoding or resp.apparent_encoding
    raw_html = resp.text
    soup = BeautifulSoup(raw_html, "html.parser")

    page = ParsedHtmlPage(
        url=target,
        final_url=str(resp.url),
        status_code=int(resp.status_code),
        soup=soup,
        raw_html=raw_html,
        encoding=encoding,
    )
    logger.info(
        "parsed html url=%s final=%s chars=%s status=%s",
        target,
        page.final_url,
        len(raw_html),
        page.status_code,
    )
    return page
