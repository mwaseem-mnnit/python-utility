"""Shared helpers for flow runners."""

from __future__ import annotations

import logging

from wix_utility.core.config import WixConfig
from wix_utility.io.csv_records import parse_csv_records

logger = logging.getLogger(__name__)


def load_optional_csv(config: WixConfig) -> list[dict[str, object]]:
    if config.input_csv is None:
        logger.warning("WIX_INPUT_CSV is not set; continuing with zero source records.")
        return []
    if not config.input_csv.is_file():
        logger.error("CSV file does not exist: %s", config.input_csv)
        return []
    return parse_csv_records(config.input_csv, delimiter=config.csv_delimiter)
