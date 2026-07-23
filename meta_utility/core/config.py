"""Environment loading and runtime settings for meta utility."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_DIR.parent
DEFAULT_LOG_DIR = WORKSPACE_ROOT / "log"

ENV_DEFAULT_JOB = "META_UTIL_DEFAULT_JOB"
ENV_LOG_DIR = "META_UTIL_LOG_DIR"

JOB_WHATSAPP_MARKETING = "whatsapp-marketing"
DEFAULT_JOB_NAME = JOB_WHATSAPP_MARKETING


def load_meta_utility_env() -> None:
    """Load ``meta_utility/.env`` for all meta utility jobs."""
    load_dotenv(PACKAGE_DIR / ".env")


def read_bool_env(key: str, *, default: bool = False) -> bool:
    """Read one environment key as boolean, fallback to default."""
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes"):
        return True
    if raw in ("0", "false", "no"):
        return False
    return default


def read_env_value(key: str, *, default: str = "", required: bool = False) -> str:
    """Read one environment key as a stripped string."""
    value = os.getenv(key, "").strip()
    if value:
        return value
    if required:
        raise ValueError(f"{key} is required in meta_utility/.env")
    return default


def read_int_env_value(key: str, *, default: int) -> int:
    """Read one environment key as integer, fallback to default."""
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def read_path_env_value(key: str, *, required: bool = False) -> Path | None:
    """Read one environment key and return a resolved path."""
    raw = read_env_value(key, required=required)
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def _csv_list_env(key: str, *, default: list[str]) -> list[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass(frozen=True)
class MetaConfig:
    graph_base_url: str
    graph_token: str
    request_timeout_seconds: int
    whatsapp_phone_number_id: str
    whatsapp_messaging_product: str
    whatsapp_recipient_type: str
    whatsapp_type: str
    whatsapp_template_name: str
    whatsapp_template_language_code: str
    whatsapp_template_component_type: str
    marketing_input_csv: Path
    marketing_csv_delimiter: str
    marketing_batch_size: int
    marketing_default_discount_amount: str
    marketing_default_min_applicable_amount: str
    marketing_default_end_date: str
    marketing_default_user_limit: str
    marketing_default_coupon_code: str
    marketing_template_parameter_keys: list[str]
    dry_run: bool


def load_meta_config() -> MetaConfig:
    """Read meta utility settings from environment variables."""
    batch_size = read_int_env_value("META_UTIL_MARKETING_BATCH_SIZE", default=50)
    input_csv = read_path_env_value("META_UTIL_MARKETING_INPUT_CSV", required=True)
    if input_csv is None:
        raise ValueError("META_UTIL_MARKETING_INPUT_CSV is required in meta_utility/.env")

    return MetaConfig(
        graph_base_url=read_env_value(
            "META_UTIL_GRAPH_BASE_URL",
            default="https://graph.facebook.com/v25.0",
        ).rstrip("/"),
        graph_token=read_env_value("META_UTIL_GRAPH_TOKEN", required=True),
        request_timeout_seconds=read_int_env_value("META_UTIL_REQUEST_TIMEOUT_SECONDS", default=30),
        whatsapp_phone_number_id=read_env_value("META_UTIL_WA_PHONE_NUMBER_ID", required=True),
        whatsapp_messaging_product=read_env_value("META_UTIL_WA_MESSAGING_PRODUCT", default="whatsapp"),
        whatsapp_recipient_type=read_env_value("META_UTIL_WA_RECIPIENT_TYPE", default="individual"),
        whatsapp_type=read_env_value("META_UTIL_WA_TYPE", default="template"),
        whatsapp_template_name=read_env_value("META_UTIL_WA_TEMPLATE_NAME", required=True),
        whatsapp_template_language_code=read_env_value("META_UTIL_WA_TEMPLATE_LANGUAGE_CODE", default="en"),
        whatsapp_template_component_type=read_env_value("META_UTIL_WA_TEMPLATE_COMPONENT_TYPE", default="body"),
        marketing_input_csv=input_csv,
        marketing_csv_delimiter=read_env_value("META_UTIL_MARKETING_CSV_DELIMITER", default=",")[:1] or ",",
        marketing_batch_size=batch_size if batch_size > 0 else 1,
        marketing_default_discount_amount=read_env_value("META_UTIL_MARKETING_DEFAULT_DISCOUNT_AMOUNT", default=""),
        marketing_default_min_applicable_amount=read_env_value(
            "META_UTIL_MARKETING_DEFAULT_MIN_APPLICABLE_AMOUNT",
            default="",
        ),
        marketing_default_end_date=read_env_value(
            "META_UTIL_MARKETING_DEFAULT_END_DATE",
            default="",
        ),
        marketing_default_user_limit=read_env_value(
            "META_UTIL_MARKETING_DEFAULT_USER_LIMIT",
            default="",
        ),
        marketing_default_coupon_code=read_env_value("META_UTIL_MARKETING_DEFAULT_COUPON_CODE", default=""),
        marketing_template_parameter_keys=_csv_list_env(
            "META_UTIL_MARKETING_TEMPLATE_PARAMETER_KEYS",
            default=["name", "discountAmount", "minApplicableAmount", "couponCode", "endDate", "userLimit"],
        ),
        dry_run=read_bool_env("META_DRY_RUN", default=True),
    )
