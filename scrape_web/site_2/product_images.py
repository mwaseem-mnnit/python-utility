from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from scrape_web.site_2.common import read_required_env
from scrape_web.site_2.html_fetch import ParsedHtmlPage

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def _image_extension(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".webp", ".jpeg", ".jpg", ".png", ".gif", ".avif", ".bmp"):
        if path.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def extract_product_images(page: ParsedHtmlPage, row: dict[str, str]) -> list[str]:
    """
    Download product images from:
    ``div.thumbnail ul li[data-type][data-src]``.

    Filenames: ``{identifier}_{idx}{ext}``, where ``idx`` starts at 0 and follows
    ``<li>`` document order.
    """
    identifier = (row.get("identifier") or "").strip()
    if not identifier:
        raise ValueError("CSV row missing identifier")

    images_dir = Path(read_required_env("SCRAPE_SITE2_IMAGES")).expanduser().resolve()
    images_dir.mkdir(parents=True, exist_ok=True)

    li_tags = page.soup.select("div.thumbnail ul li[data-src]")
    saved: list[str] = []
    base = page.final_url or page.url

    idx = 0
    for li in li_tags:
        data_type = (li.get("data-type") or "").strip().lower()
        if data_type and data_type != "img":
            continue
        src = (li.get("data-src") or "").strip()
        if not src:
            continue

        full_url = urljoin(base, src)
        ext = _image_extension(full_url)
        filename = f"{identifier}_{idx}{ext}"
        out_path = images_dir / filename

        resp = requests.get(full_url, headers=_HEADERS, timeout=60.0)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        saved.append(str(out_path))
        idx += 1

    logger.info(
        "Downloaded %s image(s) for identifier=%s into %s",
        len(saved),
        identifier,
        images_dir,
    )
    return saved
