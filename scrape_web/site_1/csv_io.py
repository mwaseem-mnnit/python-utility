from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from scrape_web.site_1.config import default_csv_path
from scrape_web.site_1.images import format_image_paths_for_csv
from scrape_web.site_1.models import ProductRow

_BASE_FIELDS = ("product_id", "product_slug", "title", "image_paths")


def product_exists(slug: str, csv_path: Path | None = None) -> bool:
    """Return True if ``product_slug`` is already present in the CSV."""
    path = csv_path if csv_path is not None else default_csv_path()
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("product_slug") == slug:
                return True
    return False


def append_products(
    rows: Iterable[ProductRow],
    csv_path: Path | None = None,
) -> None:
    """
    Merge ``rows`` into the catalogue CSV.

    Description columns ``desc_1``, ``desc_2``, ... are sized to the maximum needed
    across existing rows and new rows. The file is rewritten atomically so column
    expansion stays consistent (CSV has no native “append wider row” operation).
    """
    path = csv_path if csv_path is not None else default_csv_path()
    new_rows = list(rows)
    if not new_rows:
        return

    existing = _load_all(path)
    combined = existing + new_rows
    max_desc = _max_desc_count(combined)
    fieldnames = list(_BASE_FIELDS) + [f"desc_{i}" for i in range(1, max_desc + 1)]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in combined:
                writer.writerow(_row_to_csv_dict(row, max_desc))
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise


def _load_all(csv_path: Path) -> list[ProductRow]:
    if not csv_path.is_file():
        return []
    out: list[ProductRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        desc_keys = _desc_keys_from_fieldnames(reader.fieldnames or [])
        for raw in reader:
            if not raw.get("product_id"):
                continue
            descs: list[str] = []
            for k in desc_keys:
                descs.append((raw.get(k) or "").strip())
            while descs and descs[-1] == "":
                descs.pop()
            img = (raw.get("image_paths") or "").strip()
            paths = [p.strip() for p in img.split(",") if p.strip()]
            out.append(
                ProductRow(
                    product_id=raw["product_id"].strip(),
                    product_slug=(raw.get("product_slug") or "").strip(),
                    title=(raw.get("title") or "").strip(),
                    image_paths=paths,
                    descriptions=descs,
                )
            )
    return out


def _desc_keys_from_fieldnames(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        return []
    keys = [fn for fn in fieldnames if fn.startswith("desc_") and fn[5:].isdigit()]
    keys.sort(key=lambda x: int(x.split("_", 1)[1]))
    return keys


def _max_desc_count(rows: list[ProductRow]) -> int:
    if not rows:
        return 0
    return max(len(r.descriptions) for r in rows)


def _row_to_csv_dict(row: ProductRow, max_desc: int) -> dict[str, str]:
    d: dict[str, str] = {
        "product_id": row.product_id,
        "product_slug": row.product_slug,
        "title": row.title,
        "image_paths": format_image_paths_for_csv(row.image_paths),
    }
    for i in range(max_desc):
        key = f"desc_{i + 1}"
        d[key] = row.descriptions[i] if i < len(row.descriptions) else ""
    return d
