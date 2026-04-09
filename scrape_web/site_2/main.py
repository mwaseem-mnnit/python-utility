from __future__ import annotations

import logging
import sys

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.scrape_product import dump_product, scrape_product

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        extracted = scrape_product()
        dump_product(extracted)
        return 0
    except Exception as e:
        logger.exception("site_2 scrape_product failed: %s", e)
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
