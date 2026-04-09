from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProductRow:
    """One catalogue row: core fields plus parallel description slots (desc_1, desc_2, ... in CSV)."""

    product_id: str
    product_slug: str
    title: str
    image_paths: list[str]
    descriptions: list[str] = field(default_factory=list)
