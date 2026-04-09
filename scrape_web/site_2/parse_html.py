from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from scrape_web.site_2.html_fetch import ParsedHtmlPage

# Pagination paths like ``/product_p2``, ``/product_p3`` (page index after ``_p``).
_PAGINATION_PATH = re.compile(r"^/products_p\d+$", re.IGNORECASE)


def _pagination_path_ok(path: str) -> bool:
    return bool(_PAGINATION_PATH.match(path))


def _is_pagination_href(href: str, base: str) -> bool:
    if not href or not str(href).strip():
        return False
    full = urljoin(base, href.strip())
    path = urlparse(full).path
    return _pagination_path_ok(path)


def _normalize_site2_pagination_url(href: str, base: str) -> str | None:
    """
    Keep only canonical site pagination URLs:
    ``https://www.liuhjgled.com/products_p<number>``.
    Fragments/query are dropped.
    """
    if not href or not str(href).strip():
        return None
    full = urljoin(base, href.strip())
    parsed = urlparse(full)
    if parsed.netloc.lower() != "www.liuhjgled.com":
        return None
    if not _pagination_path_ok(parsed.path):
        return None
    return f"https://www.liuhjgled.com{parsed.path}"


def extract_product_links(page: ParsedHtmlPage) -> list[str]:
    """
    Extract product hyperlinks from each ``div.product-item``: first ``<a href="...">`` inside the div.

    Returns absolute, de-duplicated URLs preserving first-seen order.
    """
    out: list[str] = []
    seen: set[str] = set()

    for item in page.soup.find_all("div", class_="product-item"):
        a = item.find("a", href=True)
        if not a:
            continue
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(page.final_url or page.url, href)
        if full in seen:
            continue
        seen.add(full)
        out.append(full)

    return out


def extract_all_page_links(page: ParsedHtmlPage) -> list[str]:
    """
    Extract pagination page links from list markup:
    ``ul li a[href]`` where href path is exactly ``/products_p<number>``.

    Returns absolute, de-duplicated URLs preserving first-seen order.
    """
    base = page.final_url or page.url
    out: list[str] = []
    seen: set[str] = set()
    for a in page.soup.select("ul li a[href]"):
        href = (a.get("href") or "").strip()
        full = _normalize_site2_pagination_url(href, base)
        if not full:
            continue
        if full in seen:
            continue
        seen.add(full)
        out.append(full)

    return out
