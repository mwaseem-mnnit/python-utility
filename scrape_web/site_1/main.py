from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app_logging import init_logging

from scrape_web.site_1.config import get_scrape_data_dir
from scrape_web.site_1.worker import scrape_products

logger = logging.getLogger(__name__)


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
    CLI entry point: ``.env`` is loaded via ``scrape_web.site_1.config`` (on import).
    Reads ``SCRAPE_BASE_URL`` and optional ``SCRAPE_START_PAGE``, ``SCRAPE_END_PAGE``, ``SCRAPE_LIMIT``.
    """
    init_logging(default_filename="app.log")

    base_url = os.environ.get("SCRAPE_BASE_URL", "").strip()
    if not base_url:
        logger.error(
            "Missing SCRAPE_BASE_URL. Set it in scrape_web/site_1/.env (or the environment)."
        )
        return 1

    try:
        get_scrape_data_dir()
    except ValueError as e:
        logger.error("%s", e)
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
