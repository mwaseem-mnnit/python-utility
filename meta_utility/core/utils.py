"""Configuration constants and environment loading for Meta utilities."""

import hashlib
import json
from typing import Any


def payload_fingerprint(payload: Any) -> str:
    """Return a stable short identifier for request/response logging."""
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()