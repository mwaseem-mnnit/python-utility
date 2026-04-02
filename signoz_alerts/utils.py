import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    log_file = os.getenv("SIGNOZ_LOG_FILE", "signoz_alerts.log")
    if not os.path.isabs(log_file):
        log_file = os.path.join(os.getcwd(), log_file)

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root_logger.addHandler(file_handler)


def load_env_file(env_file: str = ".env") -> None:
    candidate_paths = [
        os.path.join(os.getcwd(), env_file),
        os.path.join(os.path.dirname(__file__), env_file),
    ]

    env_path = ""
    for path in candidate_paths:
        if os.path.isfile(path):
            env_path = path
            break

    if not env_path:
        return

    try:
        with open(env_path, mode="r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError as exc:
        logger.error("Failed to load env file %s: %s", env_path, exc)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def is_nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned.lower() or "unknown"


def recursive_format(data: Any, context: Dict[str, Any]) -> Any:
    if isinstance(data, dict):
        return {k: recursive_format(v, context) for k, v in data.items()}
    if isinstance(data, list):
        return [recursive_format(item, context) for item in data]
    if isinstance(data, str):
        # If this string is exactly one placeholder like "{threshold}",
        # return the underlying value from context preserving its type.
        if data.startswith("{") and data.endswith("}") and data.count("{") == 1 and data.count("}") == 1:
            key = data[1:-1]
            if key in context:
                return context[key]
        try:
            # Preserve SigNoz template variables like {{$value}} and {{$threshold}}
            # so Python format does not collapse them to single braces.
            protected = re.sub(r"\{\{\$[^{}]+\}\}", lambda m: m.group(0).replace("{", "__LBRACE__").replace("}", "__RBRACE__"), data)
            formatted = protected.format(**context)
            return formatted.replace("__LBRACE__", "{").replace("__RBRACE__", "}")
        except KeyError as exc:
            logger.error("Missing placeholder in template: %s", exc)
            return data
    return data


def payload_fingerprint(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
