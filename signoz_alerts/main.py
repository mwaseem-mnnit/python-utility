import logging
import os
import sys
from pathlib import Path
from typing import Set

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app_logging import init_logging

from alert_builder import build_alert_requests
from api_client import create_alert, fetch_existing_alert_names, skip_alert
from config import CSV_PATH
from csv_loader import load_alerts
from utils import payload_fingerprint


logger = logging.getLogger(__name__)


def _resolve_csv_path() -> str:
    return os.getenv("SIGNOZ_ALERTS_CSV_PATH", CSV_PATH)


def main() -> None:
    init_logging(default_filename="app.log")
    csv_path = _resolve_csv_path()
    logger.info("Starting SigNoz alert creation from csv=%s", csv_path)

    try:
        existing_alert_list = fetch_existing_alert_names()
        alerts = load_alerts(csv_path)
        if not alerts:
            logger.info("No valid alert rows found. Exiting.")
            return

        seen_payloads: Set[str] = set()
        created_count = 0
        skipped_duplicates = 0
        skipped_existing = 0
        failed_count = 0
        index = 0

        for alert in alerts:
            if index >= int(os.getenv("SIGNOZ_MAX_ITERATION", 1)):
                break
            index += 1
            payloads = build_alert_requests(alert)
            for payload in payloads:
                key = payload_fingerprint(payload)
                if key in seen_payloads:
                    skipped_duplicates += 1
                    logger.info("Skipping duplicate payload=%s", key[:12])
                    continue

                alert_name = str(payload.get("alert", "")).strip()
                if alert_name and skip_alert(alert_name, existing_alert_list):
                    skipped_existing += 1
                    logger.info("Skipping existing remote alert name=%s", alert_name)
                    continue

                seen_payloads.add(key)
                ok = create_alert(payload)
                if ok:
                    created_count += 1
                    if alert_name:
                        existing_alert_list.append({"id": "", "alert": alert_name})
                else:
                    failed_count += 1

        logger.info(
            "Finished alert run created=%s failed=%s skipped_duplicates=%s skipped_existing=%s total_unique=%s",
            created_count,
            failed_count,
            skipped_duplicates,
            skipped_existing,
            len(seen_payloads),
        )
    except Exception as exc:
        logger.error("Unhandled failure in alert orchestration: %s", exc)


if __name__ == "__main__":
    main()
