from __future__ import annotations

import csv
import os
from pathlib import Path
from urllib.parse import urlparse


def read_required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    raise ValueError(f"{name} is required (see scrape_web/site_2/.env).")


def slug_from_product_link(link: str) -> str:
    """
    Expected link format:
    https://www.liuhjgled.com/product/<product_slug>
    """
    path = urlparse(link).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2 and parts[0].lower() == "product":
        return parts[1].strip()
    return parts[-1].strip() if parts else ""


def site2_home_dir() -> Path:
    return Path(read_required_env("SCRAPE_SITE2_HOME")).expanduser().resolve()


def product_details_csv_path(home: Path | None = None) -> Path:
    root = home if home is not None else site2_home_dir()
    return root / "product_details.csv"


def next_csv_identifier(csv_path: Path, start_value: int) -> int:
    if not csv_path.is_file():
        return start_value

    max_id = start_value - 1
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("identifier") or "").strip()
            if not raw:
                continue
            try:
                n = int(raw)
            except ValueError:
                continue
            if n > max_id:
                max_id = n
    return max_id + 1
