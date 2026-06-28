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
