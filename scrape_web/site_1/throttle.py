from __future__ import annotations

import os
import random
import time

_DEFAULT_SECONDS = 2.0
_ENV_KEY = "SCRAPE_DELAY"
_JITTER = 0.5


def scrape_delay() -> None:
    """Sleep using ``SCRAPE_DELAY`` seconds (default 2) with uniform jitter in ``±0.5`` s."""
    raw = os.environ.get(_ENV_KEY, str(_DEFAULT_SECONDS))
    try:
        base = float(raw)
    except ValueError:
        base = _DEFAULT_SECONDS
    delta = random.uniform(-_JITTER, _JITTER)
    time.sleep(max(0.0, base + delta))


# Alias for call sites that read naturally as "wait before next request"
wait = scrape_delay
