from __future__ import annotations

import logging
import sys

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.scrape_product import dump_product, scrape_product

logger = logging.getLogger(__name__)


def scrape_and_dump_products() -> int:
    try:
        extracted = scrape_product()
        dump_product(extracted)
        return 0
    except Exception as e:
        logger.exception("site_2 scrape_product failed: %s", e)
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


def main() -> int:
    return scrape_and_dump_products()


if __name__ == "__main__":
    raise SystemExit(main())
