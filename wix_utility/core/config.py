"""Configuration constants and environment loading for Wix utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_DIR.parent
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "log"

ENV_DEFAULT_JOB = "WIX_UTIL_DEFAULT_JOB"
ENV_LOG_DIR = "WIX_UTIL_LOG_DIR"

DEFAULT_JOB_NAME = "healthcheck"
JOB_HEALTHCHECK = "healthcheck"
JOB_DRY_RUN = "dry-run"
JOB_PARSE_CSV = "parse-csv"
JOB_RUN_FLOW = "run-flow"
JOB_CREATE_COLLECTIONS = "create-collections"
JOB_COLLECTION_SYNC = "collection-sync"
JOB_PRODUCT_SYNC = "product-sync"
JOB_MEDIA_UPLOAD = "media-upload"

FLOW_COLLECTION_SYNC = "collection-sync"
FLOW_PRODUCT_SYNC = "product-sync"
FLOW_MEDIA_UPLOAD = "media-upload"
FLOW_CATALOG_SYNC = "catalog-sync"


def load_wix_utility_env() -> None:
    """Load ``wix_utility/.env`` for all Wix jobs."""
    load_dotenv(PACKAGE_DIR / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _path_env(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class WixConfig:
    """Runtime settings for Wix API jobs."""

    base_url: str
    api_key: str
    cookie: str
    site_id: str
    account_id: str
    input_csv: Path | None
    csv_output_json: Path | None
    csv_delimiter: str
    image_dir: Path | None
    output_dir: Path | None
    flow_name: str
    match_threshold: float
    collection_title_column: str
    collection_page_size: int
    collection_query_visible_only: bool
    collection_create_enabled: bool
    collection_visible: bool
    dry_run: bool
    request_timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key)


def load_wix_config() -> WixConfig:
    """Read Wix utility settings from environment variables."""
    return WixConfig(
        base_url=os.getenv("WIX_BASE_URL", "https://www.wixapis.com").rstrip("/"),
        api_key=os.getenv("WIX_API_KEY", "").strip(),
        cookie=os.getenv("WIX_COOKIE", "").strip(),
        site_id=os.getenv("WIX_SITE_ID", "").strip(),
        account_id=os.getenv("WIX_ACCOUNT_ID", "").strip(),
        input_csv=_path_env("WIX_INPUT_CSV"),
        csv_output_json=_path_env("WIX_CSV_OUTPUT_JSON"),
        csv_delimiter=os.getenv("WIX_CSV_DELIMITER", ",")[:1] or ",",
        image_dir=_path_env("WIX_IMAGE_DIR"),
        output_dir=_path_env("WIX_OUTPUT_DIR"),
        flow_name=os.getenv("WIX_FLOW", FLOW_CATALOG_SYNC).strip().lower(),
        match_threshold=_float_env("WIX_MATCH_THRESHOLD", 0.86),
        collection_title_column=os.getenv("WIX_COLLECTION_TITLE_COLUMN", "").strip(),
        collection_page_size=_int_env("WIX_COLLECTION_PAGE_SIZE", 100),
        collection_query_visible_only=_bool_env("WIX_COLLECTION_QUERY_VISIBLE_ONLY", default=True),
        collection_create_enabled=_bool_env("WIX_COLLECTION_CREATE_ENABLED", default=False),
        collection_visible=_bool_env("WIX_COLLECTION_VISIBLE", default=False),
        dry_run=_bool_env("WIX_DRY_RUN", default=True),
        request_timeout_seconds=_int_env("WIX_REQUEST_TIMEOUT_SECONDS", 30),
        max_retries=_int_env("WIX_MAX_RETRIES", 2),
        retry_backoff_seconds=_float_env("WIX_RETRY_BACKOFF_SECONDS", 1.5),
    )
