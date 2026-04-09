"""Site 1 scraper: product catalogue CSV, IDs, image paths, and scrape throttling."""

from scrape_web.site_1.csv_io import append_products, product_exists
from scrape_web.site_1.downloads import download_images
from scrape_web.site_1.http_client import REQUEST_HEADERS, fetch_html
from scrape_web.site_1.ids import next_product_id, peek_next_product_id, reset_product_id_counter
from scrape_web.site_1.images import (
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
from scrape_web.site_1.models import ProductRow
from scrape_web.site_1.parse_html import extract_product_links, parse_product_page, slug_from_product_url
from scrape_web.site_1.config import (
    PACKAGE_DIR,
    default_csv_path,
    default_images_dir,
    get_scrape_data_dir,
)
from scrape_web.site_1.throttle import scrape_delay, wait
from scrape_web.site_1.main import main
from scrape_web.site_1.worker import scrape_products

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
