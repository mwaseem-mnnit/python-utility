from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

from scraping_foundation.config import default_images_dir

# Shopify CDN image sizing (see ``context/authority.md``)
THUMBNAIL_QUERY_WIDTH = 420
PRODUCT_IMAGE_QUERY_WIDTH = 920
THUMBNAIL_SUBDIR = "thumbnail"


def ensure_absolute_http_url(url: str) -> str:
    """Turn protocol-relative ``//host/...`` into ``https://host/...`` for ``requests``."""
    u = url.strip()
    if u.startswith("//"):
        return f"https:{u}"
    return u


def with_query_width(url: str, width: int) -> str:
    """Set or replace the ``width`` query parameter (Shopify CDN)."""
    u = ensure_absolute_http_url(url)
    parsed = urlparse(u)
    pairs = dict(parse_qsl(parsed.query, keep_blank_values=True))
    pairs["width"] = str(width)
    new_query = urlencode(sorted(pairs.items()))
    return urlunparse(parsed._replace(query=new_query))


def extension_from_url(url: str) -> str:
    """File suffix from URL path (``.webp``, ``.jpg``, …); defaults to ``.jpg``."""
    path = unquote(urlparse(url).path)
    lower = path.lower()
    for ext in (".webp", ".jpeg", ".jpg", ".png", ".gif", ".avif"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def thumbnail_filename(product_id: str, url: str) -> str:
    """``<product_id>_thmb<ext>`` (authority: first gallery image is the thumbnail)."""
    return f"{product_id}_thmb{extension_from_url(url)}"


def gallery_index_filename(product_id: str, index: int, url: str) -> str:
    """``<product_id>_<index><ext>`` for gallery images after the first (index starts at 0)."""
    return f"{product_id}_{index}{extension_from_url(url)}"


def format_image_paths_for_csv(paths: list[str]) -> str:
    """
    Catalogue ``image_paths`` column: basename only per authority (no parent folders),
    comma+space separated, e.g. ``p_100001_thmb.webp, p_100001_0.webp``.
    """
    names = [Path(p.strip()).name for p in paths if p and str(p).strip()]
    return ", ".join(names)


def ensure_images_dir(directory: Path | None = None) -> Path:
    """Create the images folder (or ``directory``) if missing; return the resolved path."""
    root = directory if directory is not None else default_images_dir()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def image_filename(product_id: str, index: int) -> str:
    """``<product_id>_<index>.jpg`` (1-based index; legacy helper)."""
    if index < 1:
        raise ValueError("index must be >= 1")
    return f"{product_id}_{index}.jpg"


def default_image_path(
    product_id: str,
    index: int,
    images_dir: Path | None = None,
) -> Path:
    """Resolved path under ``images_dir`` for a generated filename."""
    base = ensure_images_dir(images_dir)
    return base / image_filename(product_id, index)
