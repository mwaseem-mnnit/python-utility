"""Small shared helpers for Wix utilities."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def payload_fingerprint(payload: Any) -> str:
    """Return a stable short identifier for request/response logging."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    """Create a URL-friendly slug from a display name."""
    lowered = value.casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    return slug.strip("-")


def dict_from_list_by_key(items: list[dict[str, Any]], key_field: str) -> dict[str, dict[str, Any]]:
    """Create a dictionary mapping key field value to item dict."""
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        key_value = item.get(key_field)
        if key_value is None:
            continue
        key_str = str(key_value).strip()
        if key_str:
            result[key_str] = item
    return result


def compute_next_offset(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    current_offset = int(metadata.get("offset") or 0)
    response_page_size = int(metadata.get("items") or 0)
    return current_offset + response_page_size


def random_decimal_between(min_value: float, max_value: float, *, precision: int = 2) -> float:
    """Return a random decimal in the inclusive range rounded to the requested precision."""
    lower = min(min_value, max_value)
    upper = max(min_value, max_value)
    return round(random.uniform(lower, upper), max(0, precision))


def random_number_between(min_value: int, max_value: int) -> int:
    """Return a random integer in the inclusive range."""
    lower = min(min_value, max_value)
    upper = max(min_value, max_value)
    return random.randint(lower, upper)


def convert_to_wix_media_url(source_url: str, *, media_width: int, media_height: int) -> str:
    """Convert a Wix static media URL into the CMS image URL format."""
    value = source_url.strip()
    if not value:
        raise ValueError("source_url is required")
    if value.startswith("wix:image://"):
        return value

    path = urlparse(value).path
    match = re.search(r"/media/(?P<media_id>.+?)(?:/v1/|$)", path)
    if match is None:
        raise ValueError(f"Unsupported Wix media URL: {source_url}")

    media_id = match.group("media_id").strip("/")
    if not media_id:
        raise ValueError(f"Unsupported Wix media URL: {source_url}")

    return (
        f"wix:image://v1/{media_id}/file.jpg"
        f"#originWidth={int(media_width)}&originHeight={int(media_height)}"
    )


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into fixed-size batches."""
    if size <= 0:
        raise ValueError("size must be greater than 0")
    return [items[index : index + size] for index in range(0, len(items), size)]
