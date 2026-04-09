from __future__ import annotations

import logging
import sys

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.scrape_product import scrape_product

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        return scrape_product()
    except Exception as e:
        logger.exception("site_2 scrape_product failed: %s", e)
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
