"""CSV parsing helpers for meta utility flows."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def read_csv_records(
    csv_path: Path,
    *,
    delimiter: str = ",",
    trim_values: bool = True,
    drop_empty_rows: bool = True,
) -> list[dict[str, Any]]:
    """Read a CSV file into dictionaries without enforcing schema."""
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            record = _normalize_row(row, trim_values=trim_values)
            if drop_empty_rows and not any(str(value).strip() for value in record.values()):
                continue
            records.append(record)
    return records


def _normalize_row(row: dict[str, str | None], *, trim_values: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = raw_key.strip()
        if not key:
            continue
        value = "" if raw_value is None else raw_value
        normalized[key] = value.strip() if trim_values else value
    return normalized

