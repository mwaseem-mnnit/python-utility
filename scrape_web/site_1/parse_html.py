from __future__ import annotations

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup


def extract_product_links(html: str, page_url: str) -> list[str]:
    """
    From ``<ul id="product-grid">``, collect ``<a href="/products/...">`` and return absolute URLs.
    Order preserved; duplicates removed while keeping first occurrence.
    """
    soup = BeautifulSoup(html, "html.parser")
    grid = soup.find("ul", id="product-grid")
    if not grid:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for a in grid.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("/products/"):
            continue
        full = urljoin(page_url, href)
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def _is_thumbnail_slider_ul(element_id: str | None) -> bool:
    if not element_id:
        return False
    return element_id.startswith("Slider-Thumbnails-template--") and element_id.endswith("__main")


def _img_src_from_tag(img, base_url: str) -> str | None:
    for attr in ("src", "data-src"):
        raw = img.get(attr)
        if not raw or not str(raw).strip():
            continue
        raw = str(raw).strip()
        if raw.startswith("data:"):
            continue
        return urljoin(base_url, raw)
    return None


def _extract_urls_from_thumbnail_slider(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Images from ``<ul id="Slider-Thumbnails-template--<epoch>__main" class="thumbnail-list ...">``
    inside ``<slide-component id="GalleryThumbnails-template--<epoch>__main">`` (see authority).
    """
    out: list[str] = []
    seen: set[str] = set()

    for ul in soup.find_all("ul", id=_is_thumbnail_slider_ul):
        parent_slider = ul.find_parent("slide-component")
        if parent_slider is None:
            continue
        slider_id = parent_slider.get("id") or ""
        if not (
            slider_id.startswith("GalleryThumbnails-template--")
            and slider_id.endswith("__main")
        ):
            continue

        for li in ul.find_all("li"):
            img = li.find("img")
            if not img:
                continue
            full = _img_src_from_tag(img, base_url)
            if full and full not in seen:
                seen.add(full)
                out.append(full)

        if out:
            return out

    # Fallback: thumbnail list id pattern without strict parent match (DOM variations)
    for ul in soup.find_all("ul", id=_is_thumbnail_slider_ul):
        for img in ul.find_all("img"):
            full = _img_src_from_tag(img, base_url)
            if full and full not in seen:
                seen.add(full)
                out.append(full)
        if out:
            return out

    return out


def parse_product_page(html: str, url: str) -> dict:
    """
    Parse a product detail page. Returns keys: ``product_slug``, ``title``,
    ``descriptions`` (list of ``<p>`` texts), ``image_urls`` (ordered URLs from the
    thumbnail slider per ``context/authority.md``).
    """
    soup = BeautifulSoup(html, "html.parser")
    product_slug = slug_from_product_url(url)

    title_el = soup.select_one("div.product__title h1")
    title = title_el.get_text(strip=True) if title_el else ""

    desc_root = soup.select_one("div.product__description.rte.quick-add-hidden")
    descriptions: list[str] = []
    if desc_root:
        for p in desc_root.find_all("p"):
            t = p.get_text(strip=True)
            if t:
                descriptions.append(t)

    image_urls = _extract_urls_from_thumbnail_slider(soup, url)

    return {
        "product_slug": product_slug,
        "title": title,
        "descriptions": descriptions,
        "image_urls": image_urls,
    }


def slug_from_product_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "products":
        return parts[1]
    return parts[-1] if parts else ""
