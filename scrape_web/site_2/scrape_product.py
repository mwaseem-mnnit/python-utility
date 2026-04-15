from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import logging
import os
from pathlib import Path

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.common import product_details_csv_path, site2_home_dir
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
    """Load ``product_details.csv`` rows; if a ``scrape`` column exists, keep only ``scrape == 0``."""
    path = product_details_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"product_details.csv not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_scrape = "scrape" in fieldnames
        for row in reader:
            if not row:
                continue
            cleaned = {k: (v or "").strip() for k, v in row.items() if k}
            if has_scrape and cleaned.get("scrape") == "0":
                continue
            rows.append(cleaned)
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


def scrape_product() -> list[ProductRowDetail]:
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

    logger.info("Extracted product detail rows=%s", len(extracted))
    return extracted


_NS_PRODUCT_FIELDNAMES = [
    "handleId",
    "brand",
    "name",
    "description",
    "additionalInfoTitle1",
    "additionalInfoDescription1",
    "additionalInfoTitle2",
    "additionalInfoDescription2",
    "fieldType",
    "sku",
    "price",
    "visible",
    "inventory",
]


def _product_row_csv_dict(item: ProductRowDetail) -> dict[str, str]:
    data = asdict(item)
    return {
        "handleId": data["identifier"],
        "brand": data["brand"],
        "name": data["title"],
        "description": data["description"],
        "additionalInfoTitle1": data["additionalInfoTitle1"],
        "additionalInfoDescription1": data["additionalInfoDescription1"],
        "additionalInfoTitle2": data["additionalInfoTitle2"],
        "additionalInfoDescription2": data["additionalInfoDescription2"],
        "fieldType": "Product",
        "sku": "100",
        "price": "1000",
        "visible": "FALSE",
        "inventory": "InStock",
    }


def dump_product(extracted: list[ProductRowDetail]) -> Path:
    """
    Write extracted products to
    ``<SCRAPE_SITE2_HOME>/ns_product_<dd-mm-yyyy>.csv``.

    If the file already exists, rows with the same ``handleId`` are replaced;
    other rows are kept. New ``handleId`` values are appended.
    """
    home = site2_home_dir()
    home.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%d-%m-%Y")
    out_path = home / f"ns_product_{date_str}.csv"

    fieldnames = _NS_PRODUCT_FIELDNAMES
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []

    if out_path.is_file():
        with out_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                if not raw:
                    continue
                hid = (raw.get("handleId") or "").strip()
                if not hid:
                    continue
                row = {k: (raw.get(k) or "").strip() for k in fieldnames}
                if hid not in merged:
                    order.append(hid)
                merged[hid] = row

    for item in extracted:
        row = _product_row_csv_dict(item)
        hid = row["handleId"]
        if hid not in merged:
            order.append(hid)
        merged[hid] = row

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for hid in order:
            writer.writerow(merged[hid])

    logger.info(
        "Dumped product rows=%s (merged) to %s",
        len(order),
        out_path,
    )
    print(len(extracted))
    return out_path
