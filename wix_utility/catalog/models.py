"""Small data models used before payloads are mapped to Wix API schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProductImage:
    """Local image file intended for upload and association with a Wix product."""

    path: Path
    alt_text: str = ""
    wix_media_id: str = ""


@dataclass(frozen=True)
class CollectionDraft:
    """Collection/category data before conversion to a Wix API payload."""

    name: str
    slug: str = ""
    description: str = ""
    visible: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductDraft:
    """Product data from CSV/chat parsing before conversion to Wix API payloads."""

    name: str
    sku: str = ""
    description: str = ""
    price: float | None = None
    collections: tuple[str, ...] = ()
    images: tuple[ProductImage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
