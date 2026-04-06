from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
load_dotenv(PACKAGE_DIR / ".env")


def get_scrape_data_dir() -> Path:
    """
    Root folder for catalogue data (CSV and ``images/`` subfolder), from ``SCRAPE_DATA_DIR``.
    """
    raw = os.environ.get("SCRAPE_DATA_DIR", "").strip()
    if not raw:
        raise ValueError(
            "SCRAPE_DATA_DIR is not set. Add it to scraping_foundation/.env "
            "(absolute path to the folder that will hold products.csv and images/)."
        )
    return Path(raw).expanduser().resolve()


def default_csv_path() -> Path:
    """``<SCRAPE_DATA_DIR>/products.csv``."""
    return get_scrape_data_dir() / "products.csv"


def default_images_dir() -> Path:
    """``<SCRAPE_DATA_DIR>/images``."""
    return get_scrape_data_dir() / "images"
