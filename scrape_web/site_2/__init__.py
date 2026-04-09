"""Site 2 scraper package."""

from scrape_web.site_2.html_fetch import ParsedHtmlPage, fetch_and_parse_html
from scrape_web.site_2.parse_html import extract_all_page_links, extract_product_links

__all__ = [
    "ParsedHtmlPage",
    "extract_all_page_links",
    "extract_product_links",
    "fetch_and_parse_html",
]
