from __future__ import annotations

import csv
import logging
from collections import deque
from pathlib import Path

from scrape_web.site_2.parse_html import extract_all_page_links, extract_product_links
from scrape_web.site_2.common import (
    next_csv_identifier,
    product_details_csv_path,
    read_required_env,
    site2_home_dir,
    slug_from_product_link,
)
from scrape_web.site_2.html_fetch import fetch_and_parse_html

logger = logging.getLogger(__name__)


def crawl_all_product_links(start_url: str) -> list[str]:
    """
    Start from ``start_url``, discover pagination links on each page, and
    collect product links from every visited page.
    """
    queue: deque[str] = deque([start_url])
    seen_pages: set[str] = set()
    seen_products: set[str] = set()
    products: list[str] = []

    while queue:
        page_url = queue.popleft()
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)
        logger.info("Visiting page %s", page_url)

        page = fetch_and_parse_html(page_url)

        page_product_links = extract_product_links(page)
        for link in page_product_links:
            if link in seen_products:
                continue
            seen_products.add(link)
            products.append(link)

        discovered_page_links = extract_all_page_links(page)
        for next_page in discovered_page_links:
            if next_page not in seen_pages:
                queue.append(next_page)

        logger.info(
            "Page done %s: products_found=%s, pages_discovered=%s, queue=%s",
            page_url,
            len(page_product_links),
            len(discovered_page_links),
            len(queue),
        )

    return products


def write_product_details_csv(
    *,
    home_dir: Path,
    product_links: list[str],
    identifier_start: int,
) -> Path:
    home_dir.mkdir(parents=True, exist_ok=True)
    csv_path = product_details_csv_path(home_dir)
    fieldnames = ["identifier", "slug", "link"]
    first_id = next_csv_identifier(csv_path, identifier_start)

    file_exists = csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        identifier = first_id
        for link in product_links:
            slug = slug_from_product_link(link)
            writer.writerow(
                {
                    "identifier": identifier,
                    "slug": slug,
                    "link": link,
                }
            )
            identifier += 1

    return csv_path


def run_site2_scrape() -> int:
    start_url = read_required_env("SCRAPE_SITE2_URL")
    home = site2_home_dir()
    identifier_start = int(read_required_env("SCRAPE_SITE2_IDENTIFIER_START"))

    product_links = crawl_all_product_links(start_url)
    csv_path = write_product_details_csv(
        home_dir=home,
        product_links=product_links,
        identifier_start=identifier_start,
    )

    msg = (
        f"OK pages crawled from {start_url!r}; "
        f"products={len(product_links)}; csv={str(csv_path)!r}"
    )
    logger.info(msg)
    return 0
