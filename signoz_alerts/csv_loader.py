import csv
import logging
import os
from typing import Dict, List

from utils import is_nonempty


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"service_name", "http_route", "p99", "p50", "error_rate"}


def load_alerts(csv_path: str) -> List[Dict]:
    if not is_nonempty(csv_path):
        logger.error("CSV path is empty.")
        return []
    if not os.path.isfile(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        return []

    rows: List[Dict] = []
    try:
        with open(csv_path, mode="r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                logger.error("CSV has no header row: %s", csv_path)
                return []

            header = {col.strip() for col in reader.fieldnames}
            missing = REQUIRED_COLUMNS - header
            if missing:
                logger.error("CSV missing required columns: %s", sorted(missing))
                return []

            for idx, row in enumerate(reader, start=2):
                service_name = (row.get("service_name") or "").strip()
                http_route = (row.get("http_route") or "").strip()

                if not service_name or not http_route:
                    logger.error(
                        "Skipping row %s due to missing service_name/http_route", idx
                    )
                    continue

                rows.append(
                    {
                        "service_name": service_name,
                        "http_route": http_route,
                        "p99": (row.get("p99") or "").strip(),
                        "p50": (row.get("p50") or "").strip(),
                        "error_rate": (row.get("error_rate") or "").strip(),
                    }
                )
    except Exception as exc:
        logger.error("Failed reading CSV %s: %s", csv_path, exc)
        return []

    return rows
