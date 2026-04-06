from __future__ import annotations

import logging
from pathlib import Path

import requests

from scraping_foundation.http_client import REQUEST_HEADERS
from scraping_foundation.images import (
    PRODUCT_IMAGE_QUERY_WIDTH,
    THUMBNAIL_QUERY_WIDTH,
    THUMBNAIL_SUBDIR,
    ensure_images_dir,
    gallery_index_filename,
    thumbnail_filename,
    with_query_width,
)

logger = logging.getLogger(__name__)


def _relative_or_absolute(dest: Path) -> str:
    try:
        return str(dest.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(dest)


def _fetch_to(url: str, dest: Path, *, timeout: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout, stream=True)
    r.raise_for_status()
    dest.write_bytes(r.content)


def download_images(
    image_urls: list[str],
    product_id: str,
    images_dir: Path | None = None,
    *,
    timeout: float = 60.0,
) -> list[str]:
    """
    Per ``context/authority.md``:

    * First list item → ``<thumbnail>/<product_id>_thmb.<ext>`` using ``width=420``.
    * Every item (including the first) → ``<product_id>_<index>.<ext>`` at gallery root
      using ``width=920``, indices ``0 .. n-1``.

    Returns paths in order: thumbnail file, then ``_0``, ``_1``, …
    """
    root = ensure_images_dir(images_dir)
    thumb_dir = root / THUMBNAIL_SUBDIR
    saved: list[str] = []

    if not image_urls:
        return saved

    # Thumbnail: first item only, width=420, under thumbnail/
    u_thumb = with_query_width(image_urls[0], THUMBNAIL_QUERY_WIDTH)
    dest_thumb = thumb_dir / thumbnail_filename(product_id, image_urls[0])
    try:
        _fetch_to(u_thumb, dest_thumb, timeout=timeout)
    except (requests.RequestException, OSError) as e:
        logger.error("failed to download %s -> %s: %s", u_thumb, dest_thumb, e)
        raise
    saved.append(_relative_or_absolute(dest_thumb))

    # Product images: all items including first, width=920, index from 0
    for i, src in enumerate(image_urls):
        u = with_query_width(src, PRODUCT_IMAGE_QUERY_WIDTH)
        dest = root / gallery_index_filename(product_id, i, src)
        try:
            _fetch_to(u, dest, timeout=timeout)
        except (requests.RequestException, OSError) as e:
            logger.error("failed to download %s -> %s: %s", u, dest, e)
            raise
        saved.append(_relative_or_absolute(dest))

    return saved
