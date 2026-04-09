from __future__ import annotations

import logging
import os
from pathlib import Path
from scrape_web.site_1.csv_io import append_products, product_exists
from scrape_web.site_1.downloads import download_images
from scrape_web.site_1.http_client import fetch_html
from scrape_web.site_1.ids import next_product_id
from scrape_web.site_1.models import ProductRow
from scrape_web.site_1.parse_html import extract_product_links, parse_product_page, slug_from_product_url
from scrape_web.site_1.config import default_csv_path, default_images_dir
from scrape_web.site_1.throttle import wait

logger = logging.getLogger(__name__)


def _catalog_page_url(base: str, page: int) -> str:
    base = base.rstrip("/")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}page={page}"


def _resolve_base_url(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    env = os.environ.get("SCRAPE_BASE_URL", "").strip()
    if env:
        return env.rstrip("/")
    raise ValueError("Pass base_url= to scrape_products() or set SCRAPE_BASE_URL")


def scrape_products(
    *,
    base_url: str | None = None,
    start_page: int = 1,
    end_page: int = 16,
    limit: int | None = None,
    csv_path: Path | None = None,
    images_dir: Path | None = None,
) -> None:
    """
    Crawl catalogue pages ``?page=`` … extract product links, then for each product
    fetch detail HTML (with :func:`wait` before each product request), parse, download
    images, and append rows to the CSV.

    * ``base_url`` — listing URL without ``page=`` (e.g. collection URL). Falls back to
      ``SCRAPE_BASE_URL``.
    * ``limit`` — stop after this many product queue entries (including skips), for tests.
    """
    base = _resolve_base_url(base_url)
    csv_p = csv_path if csv_path is not None else default_csv_path()
    img_dir = images_dir if images_dir is not None else default_images_dir()

    seen_links: set[str] = set()
    handled = 0

    for page in range(start_page, end_page + 1):
        if limit is not None and handled >= limit:
            break
        page_url = _catalog_page_url(base, page)
        try:
            listing_html = fetch_html(page_url)
        except Exception as e:
            logger.error("[page %s] fetch failed %s: %s", page, page_url, e)
            raise

        links = extract_product_links(listing_html, page_url)
        logger.info("[page %s] found %s product link(s)", page, len(links))

        for link in links:
            if limit is not None and handled >= limit:
                return
            if link in seen_links:
                continue
            seen_links.add(link)

            slug = slug_from_product_url(link)
            if not slug:
                logger.info("[page %s] skip (no slug) %s", page, link)
                handled += 1
                continue

            if product_exists(slug, csv_p):
                logger.info("[page %s] skipped (exists) %s", page, slug)
                handled += 1
                continue

            wait()
            try:
                product_html = fetch_html(link)
            except Exception as e:
                logger.error("[page %s] product fetch failed %s: %s", page, slug, e)
                raise

            data = parse_product_page(product_html, link)
            pid = next_product_id()
            paths: list[str] = []
            if data["image_urls"]:
                paths = download_images(data["image_urls"], pid, img_dir)

            row = ProductRow(
                product_id=pid,
                product_slug=data["product_slug"] or slug,
                title=data["title"],
                image_paths=paths,
                descriptions=data["descriptions"],
            )
            append_products([row], csv_p)
            logger.info("[page %s] processed %s", page, slug)
            handled += 1
