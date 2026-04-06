from __future__ import annotations

import os
from pathlib import Path
from scraping_foundation.csv_io import append_products, product_exists
from scraping_foundation.downloads import download_images
from scraping_foundation.http_client import fetch_html
from scraping_foundation.ids import next_product_id
from scraping_foundation.models import ProductRow
from scraping_foundation.parse_html import extract_product_links, parse_product_page, slug_from_product_url
from scraping_foundation.config import default_csv_path, default_images_dir
from scraping_foundation.throttle import wait


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
            print(f"[page {page}] fetch failed {page_url}: {e}")
            raise

        links = extract_product_links(listing_html, page_url)
        print(f"[page {page}] found {len(links)} product link(s)")

        for link in links:
            if limit is not None and handled >= limit:
                return
            if link in seen_links:
                continue
            seen_links.add(link)

            slug = slug_from_product_url(link)
            if not slug:
                print(f"[page {page}] skip (no slug) {link}")
                handled += 1
                continue

            if product_exists(slug, csv_p):
                print(f"[page {page}] skipped (exists) {slug}")
                handled += 1
                continue

            wait()
            try:
                product_html = fetch_html(link)
            except Exception as e:
                print(f"[page {page}] product fetch failed {slug}: {e}")
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
            print(f"[page {page}] processed {slug}")
            handled += 1
