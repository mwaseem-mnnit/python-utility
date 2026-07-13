"""Configuration constants and environment loading for Wix utilities."""

from __future__ import annotations

import os
import re
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
JOB_ASSIGN_COLLECTION_TO_PRODUCT = "assign-collection-to-product"
JOB_COLLECTION_PRODUCTS_TO_CMS = "collection-products-to-cms"
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


def _env_first(*names: str) -> str:
    for name in names:
        raw = os.getenv(name, "").strip()
        if raw:
            return raw
    return ""


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


def _float_env_first(default: float, *names: str) -> float:
    raw = _env_first(*names)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env_first(default: int, *names: str) -> int:
    raw = _env_first(*names)
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def _csv_list_env(*names: str) -> list[str]:
    raw = _env_first(*names)
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[,;\n]+", raw) if part.strip()]


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
    product_page_size: int
    filter_collection_id_list: list[str]
    target_cms_table_id: str
    batch_size: int
    cms_record_version: float
    precision: int
    min_rating: float
    max_rating: float
    review_count_from: int
    review_count_to: int
    media_width: int
    media_height: int
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
        product_page_size=_int_env("WIX_PRODUCT_PAGE_SIZE", 100),
        filter_collection_id_list=_csv_list_env("filterCollectionIdList", "WIX_FILTER_COLLECTION_ID_LIST"),
        target_cms_table_id=_env_first("targetCMSTableId", "WIX_TARGET_CMS_TABLE_ID"),
        batch_size=_int_env_first(100, "batchSize", "WIX_BATCH_SIZE"),
        cms_record_version=_float_env_first(1.0, "cmsRecordVersion", "WIX_CMS_RECORD_VERSION"),
        precision=_int_env_first(2, "precision", "WIX_PRECISION"),
        min_rating=_float_env_first(1.0, "minRating", "WIX_MIN_RATING"),
        max_rating=_float_env_first(5.0, "maxRating", "WIX_MAX_RATING"),
        review_count_from=_int_env_first(0, "reviewCountFrom", "WIX_REVIEW_COUNT_FROM"),
        review_count_to=_int_env_first(1000, "reviewCountTo", "WIX_REVIEW_COUNT_TO"),
        media_width=_int_env_first(50, "mediaWidth", "WIX_MEDIA_WIDTH"),
        media_height=_int_env_first(50, "mediaHeight", "WIX_MEDIA_HEIGHT"),
        collection_query_visible_only=_bool_env("WIX_COLLECTION_QUERY_VISIBLE_ONLY", default=True),
        collection_create_enabled=_bool_env("WIX_COLLECTION_CREATE_ENABLED", default=False),
        collection_visible=_bool_env("WIX_COLLECTION_VISIBLE", default=False),
        dry_run=_bool_env("WIX_DRY_RUN", default=True),
        request_timeout_seconds=_int_env("WIX_REQUEST_TIMEOUT_SECONDS", 30),
        max_retries=_int_env("WIX_MAX_RETRIES", 2),
        retry_backoff_seconds=_float_env("WIX_RETRY_BACKOFF_SECONDS", 1.5),
    )
