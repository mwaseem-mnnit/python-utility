from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProductRowDetail:
    """One product row from CSV plus fields scraped from the product page."""

    identifier: str
    slug: str
    link: str
    brand: str
    title: str
    description: str
    additionalInfoTitle1: str
    additionalInfoDescription1: str
    additionalInfoTitle2: str
    additionalInfoDescription2: str
