from __future__ import annotations

import csv
import logging
import os

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.common import product_details_csv_path
from scrape_web.site_2.html_fetch import fetch_and_parse_html
from scrape_web.site_2.product_extractors import build_product_row_detail
from scrape_web.site_2.product_images import extract_product_images
from scrape_web.site_2.product_models import ProductRowDetail

logger = logging.getLogger(__name__)


def _max_iteration() -> int | None:
    raw = os.environ.get("SCRAPE_SITE2_MAX_ITERATION", "").strip()
    if not raw:
        return None
    return int(raw)


def load_product_detail_rows() -> list[dict[str, str]]:
    path = product_details_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"product_details.csv not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append({k: (v or "").strip() for k, v in row.items() if k})
    return rows


def scrape_one_product_row(row: dict[str, str]) -> ProductRowDetail:
    identifier = (row.get("identifier") or "").strip()
    slug = (row.get("slug") or "").strip()
    link = (row.get("link") or "").strip()
    if not link:
        raise ValueError("CSV row missing link")

    page = fetch_and_parse_html(link)
    extract_product_images(page, row)
    return build_product_row_detail(
        identifier=identifier,
        slug=slug,
        link=link,
        soup=page.soup,
    )


def scrape_product() -> int:
    rows = load_product_detail_rows()
    max_iter = _max_iteration()
    extracted: list[ProductRowDetail] = []

    for i, row in enumerate(rows, start=1):
        if max_iter is not None and i > max_iter:
            logger.info(
                "Stopping at SCRAPE_SITE2_MAX_ITERATION=%s (processed %s row(s)).",
                max_iter,
                len(extracted),
            )
            break
        logger.info("Product row %s/%s link=%s", i, len(rows), row.get("link"))
        detail = scrape_one_product_row(row)
        extracted.append(detail)

    print(len(extracted))
    return 0
