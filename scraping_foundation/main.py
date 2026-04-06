from __future__ import annotations

import os
import sys

from scraping_foundation.config import get_scrape_data_dir
from scraping_foundation.worker import scrape_products


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return int(raw)


def main() -> int:
    """
    CLI entry point: ``.env`` is loaded via ``scraping_foundation.config`` (on import).
    Reads ``SCRAPE_BASE_URL`` and optional ``SCRAPE_START_PAGE``, ``SCRAPE_END_PAGE``, ``SCRAPE_LIMIT``.
    """
    base_url = os.environ.get("SCRAPE_BASE_URL", "").strip()
    if not base_url:
        print(
            "Missing SCRAPE_BASE_URL. Set it in scraping_foundation/.env (or the environment).",
            file=sys.stderr,
        )
        return 1

    try:
        get_scrape_data_dir()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    start_page = _env_int("SCRAPE_START_PAGE", 1)
    end_page = _env_int("SCRAPE_END_PAGE", 16)
    limit = _env_optional_int("SCRAPE_LIMIT")

    scrape_products(
        base_url=base_url,
        start_page=start_page,
        end_page=end_page,
        limit=limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
