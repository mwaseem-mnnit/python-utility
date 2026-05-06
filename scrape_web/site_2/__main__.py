"""Run with: ``python -m scrape_web.site_2``."""

from __future__ import annotations

import os
import sys

import scrape_web.site_2.config  # noqa: F401 — env + logging
from scrape_web.site_2.main import scrape_and_dump_products
from scrape_web.site_2.run_site2 import collect_all_product_page_links


def main() -> int:
    flow = os.environ.get("SCRAPE_SITE2_FLOW", "scrape_and_dump").strip().lower()
    workflows = {
        "scrape_and_dump": scrape_and_dump_products,
        "collect_product_links": collect_all_product_page_links,
    }
    runner = workflows.get(flow)
    if runner is None:
        available = ", ".join(sorted(workflows))
        print(
            f"Unsupported SCRAPE_SITE2_FLOW={flow!r}. Supported: {available}",
            file=sys.stderr,
        )
        return 1
    return runner()


raise SystemExit(main())
