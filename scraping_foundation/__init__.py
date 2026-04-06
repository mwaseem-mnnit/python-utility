"""Scraping foundation: product catalogue CSV, IDs, image paths, and scrape throttling."""

from scraping_foundation.csv_io import append_products, product_exists
from scraping_foundation.downloads import download_images
from scraping_foundation.http_client import REQUEST_HEADERS, fetch_html
from scraping_foundation.ids import next_product_id, peek_next_product_id, reset_product_id_counter
from scraping_foundation.images import (
    PRODUCT_IMAGE_QUERY_WIDTH,
    THUMBNAIL_QUERY_WIDTH,
    default_image_path,
    ensure_absolute_http_url,
    ensure_images_dir,
    extension_from_url,
    format_image_paths_for_csv,
    gallery_index_filename,
    image_filename,
    thumbnail_filename,
    with_query_width,
)
from scraping_foundation.models import ProductRow
from scraping_foundation.parse_html import extract_product_links, parse_product_page, slug_from_product_url
from scraping_foundation.config import (
    PACKAGE_DIR,
    default_csv_path,
    default_images_dir,
    get_scrape_data_dir,
)
from scraping_foundation.throttle import scrape_delay, wait
from scraping_foundation.main import main
from scraping_foundation.worker import scrape_products

__all__ = [
    "PRODUCT_IMAGE_QUERY_WIDTH",
    "PACKAGE_DIR",
    "THUMBNAIL_QUERY_WIDTH",
    "default_csv_path",
    "default_images_dir",
    "get_scrape_data_dir",
    "REQUEST_HEADERS",
    "ProductRow",
    "append_products",
    "default_image_path",
    "download_images",
    "ensure_absolute_http_url",
    "ensure_images_dir",
    "extension_from_url",
    "extract_product_links",
    "format_image_paths_for_csv",
    "gallery_index_filename",
    "fetch_html",
    "image_filename",
    "main",
    "next_product_id",
    "parse_product_page",
    "peek_next_product_id",
    "product_exists",
    "reset_product_id_counter",
    "scrape_delay",
    "scrape_products",
    "slug_from_product_url",
    "thumbnail_filename",
    "wait",
    "with_query_width",
]
