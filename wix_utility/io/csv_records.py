"""CSV parsing helpers for catalog import flows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def parse_csv_records(
    csv_path: Path,
    *,
    delimiter: str = ",",
    trim_values: bool = True,
    drop_empty_rows: bool = True,
) -> list[dict[str, Any]]:
    """Read a CSV file into JSON-friendly dictionaries."""
    records: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            record = _normalize_row(row, trim_values=trim_values)
            if drop_empty_rows and not any(str(value).strip() for value in record.values()):
                continue
            records.append(record)
    return records


def write_json_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write parsed CSV records as pretty JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _normalize_row(row: dict[str, str | None], *, trim_values: bool) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in row.items():
        if raw_key is None:
            continue
        key = raw_key.strip()
        if not key:
            continue
        value = "" if raw_value is None else raw_value
        if trim_values:
            value = value.strip()
        normalized[key] = _coerce_scalar(value)
    return normalized


def _coerce_scalar(value: str) -> Any:
    """Convert common CSV scalar strings while preserving product codes."""
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if not value or value.startswith("0"):
        return value
    try:
        return int(value, 10)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
