"""Small shared helpers for Wix utilities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


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
    """Create a dictionary mapping key field value to item dict for each item in the list.
    
    If key field is missing or empty, the item is skipped.
    """
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        key_value = item.get(key_field)
        if key_value is None:
            continue
        key_str = str(key_value).strip()
        if key_str:
            result[key_str] = item
    return result

def _compute_next_offset(payload: dict[str, Any]) -> int:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    current_offset = int(metadata.get("offset") or 0)
    response_page_size = int(metadata.get("items") or 0)
    return current_offset + response_page_size
