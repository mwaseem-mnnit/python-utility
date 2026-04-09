from __future__ import annotations

import re

from bs4 import BeautifulSoup

from scrape_web.site_2.product_models import ProductRowDetail


def extract_brand(soup: BeautifulSoup) -> str:
    """Value associated with ``<span>Brand</span>`` (next text/sibling)."""
    for span in soup.find_all("span"):
        if span.get_text(strip=True) != "Brand":
            continue
        for sib in span.next_siblings:
            name = getattr(sib, "name", None)
            if name in ("span", "td", "div", "strong", "b"):
                return sib.get_text(strip=True)
            if isinstance(sib, str):
                t = sib.strip()
                if t:
                    return t
        parent = span.parent
        if parent:
            text = parent.get_text(strip=True)
            return text.replace("Brand", "", 1).strip()
    return ""


def extract_alcet_title(soup: BeautifulSoup) -> str:
    """Text from ``div.alcet_title``."""
    div = soup.find("div", class_="alcet_title")
    return div.get_text(strip=True) if div else ""


def extract_description_before_share(soup: BeautifulSoup) -> str:
    """The ``<p>`` immediately before the AddThis share toolbox div."""
    share = soup.find("div", class_=re.compile(r"addthis_inline_share_toolbox"))
    if not share:
        return ""
    prev = share.find_previous("p")
    return prev.get_text(strip=True) if prev else ""


def extract_nei_table_html(soup: BeautifulSoup) -> str:
    """Full inner HTML of ``div.nei-table`` (table block)."""
    div = soup.find("div", class_="nei-table")
    return str(div) if div else ""


def extract_features_html(soup: BeautifulSoup) -> str:
    """
    ``<p>`` tags after ``div.alcet_title`` and before the warranty span,
    excluding paragraphs inside ``div.nei-table``.
    """
    alcet = soup.find("div", class_="alcet_title")
    warranty = soup.find("span", string=re.compile(r"Warranty Procedures", re.I))
    if not warranty:
        warranty = soup.find("span", string=lambda t: t and "Warranty Procedures" in t)
    if not alcet or not warranty:
        return ""

    parts: list[str] = []
    for el in alcet.find_all_next():
        if el is warranty:
            break
        if getattr(el, "name", None) != "p":
            continue
        if any(p is alcet for p in el.parents):
            continue
        if el.find_parent("div", class_="nei-table"):
            continue
        parts.append(str(el))
    return "".join(parts)


def build_product_row_detail(
    *,
    identifier: str,
    slug: str,
    link: str,
    soup: BeautifulSoup,
) -> ProductRowDetail:
    return ProductRowDetail(
        identifier=identifier,
        slug=slug,
        link=link,
        brand=extract_brand(soup),
        title=extract_alcet_title(soup),
        description=extract_description_before_share(soup),
        additionalInfoTitle1="Product Description",
        additionalInfoDescription1=extract_nei_table_html(soup),
        additionalInfoTitle2="Features",
        additionalInfoDescription2=extract_features_html(soup),
    )
