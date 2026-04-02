from __future__ import annotations

import logging
from typing import Dict, List, Set

from config import (
    DEFAULT_CPU_THRESHOLD,
    DEFAULT_ERROR_RATE_THRESHOLD,
    DEFAULT_MEMORY_THRESHOLD_MBY,
    DEFAULT_P50_THRESHOLD_MS,
    DEFAULT_P99_THRESHOLD_MS,
    DEFAULT_PREFERRED_CHANNEL,
    ERROR_RATE_THRESHOLD_INCREASE_PERCENT,
    LATENCY_THRESHOLD_INCREASE_PERCENT,
    SIGNOZ_BASE_URL,
    get_template,
)
from utils import recursive_format, sanitize_name, to_float


logger = logging.getLogger(__name__)

_resource_alert_services: Set[str] = set()


def _apply_percentage_increase(value: float, increase_percent: float) -> float:
    return value * (1 + (increase_percent / 100.0))


def _parse_duration_ms(raw_value: object) -> float | None:
    """
    Parse duration strings like "250ms", "1.2s", "300" into milliseconds.
    - If no unit is provided, treat as milliseconds.
    - Returns None for invalid/empty values.
    """
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if not text:
        return None

    if text.endswith("ms"):
        num = to_float(text[:-2].strip())
        return num if num is not None else None

    if text.endswith("s"):
        num = to_float(text[:-1].strip())
        return num * 1000.0 if num is not None else None

    return to_float(text)


def _parse_error_rate(raw_value: object) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    if text.lower() in {"na", "n/a", "null", "none", "-"}:
        return None

    if text.endswith("%"):
        numeric = to_float(text[:-1].strip())
        if numeric is None:
            return None
        return numeric / 100.0

    return to_float(text)


def compute_latency_threshold(metric_key: str, raw_value: object) -> float:
    floor_ms = DEFAULT_P99_THRESHOLD_MS if metric_key == "p99" else DEFAULT_P50_THRESHOLD_MS
    parsed = _parse_duration_ms(raw_value)
    if parsed is None or parsed < floor_ms:
        return floor_ms
    return _apply_percentage_increase(parsed, LATENCY_THRESHOLD_INCREASE_PERCENT)


def compute_error_rate_threshold(raw_value: object) -> float:
    cap = DEFAULT_ERROR_RATE_THRESHOLD
    parsed = _parse_error_rate(raw_value)
    if parsed is None or parsed <= 0:
        return cap
    bumped = _apply_percentage_increase(parsed, ERROR_RATE_THRESHOLD_INCREASE_PERCENT)
    return min(bumped, cap)


def _base_context(alert_obj: Dict) -> Dict:
    service_name = str(alert_obj["service_name"]).strip()
    http_route = str(alert_obj["http_route"]).strip()
    route_name = http_route
    return {
        "service_name": service_name,
        "route_name": route_name,
        "route_slug": sanitize_name(route_name),
        "SIGNOZ_PREFERRED_CHANNEL": DEFAULT_PREFERRED_CHANNEL,
        "SIGNOZ_BASE_URL": SIGNOZ_BASE_URL,
    }


def _build_latency_payload(metric_key: str, threshold: float, context: Dict) -> Dict:
    template = get_template("latency")
    space_aggregation = "p99" if metric_key == "p99" else "p50"
    per_metric_context = {
        **context,
        "metric_key": metric_key,
        "threshold": threshold,
        "space_aggregation": space_aggregation,
    }
    return recursive_format(template, per_metric_context)


def _build_error_rate_payload(threshold: float, context: Dict) -> Dict:
    template = get_template("error_rate")
    return recursive_format(template, {**context, "threshold": threshold, "metric_key": "error_rate"})


def _build_resource_payload(service_name: str, resource_metric: str) -> Dict:
    if resource_metric == "cpu":
        template_key = "cpu_high"
        threshold = DEFAULT_CPU_THRESHOLD
    else:
        template_key = "memory_high"
        threshold = DEFAULT_MEMORY_THRESHOLD_MBY

    template = get_template(template_key)
    context = {
        "service_name": service_name,
        "metric_key": resource_metric,
        "threshold": threshold,
        "SIGNOZ_PREFERRED_CHANNEL": DEFAULT_PREFERRED_CHANNEL,
        "SIGNOZ_BASE_URL": SIGNOZ_BASE_URL,
    }
    return recursive_format(template, context)


def build_alert_requests(alert_obj: Dict) -> List[Dict]:
    payloads: List[Dict] = []
    context = _base_context(alert_obj)

    for metric_key in ("p99", "p50"):
        threshold = compute_latency_threshold(metric_key, alert_obj.get(metric_key))
        payloads.append(_build_latency_payload(metric_key, threshold, context))

    payloads.append(
        _build_error_rate_payload(
            compute_error_rate_threshold(alert_obj.get("error_rate")), context
        )
    )

    service_name = context["service_name"]
    if service_name not in _resource_alert_services:
        payloads.append(_build_resource_payload(service_name, "cpu"))
        payloads.append(_build_resource_payload(service_name, "memory"))
        _resource_alert_services.add(service_name)

    return payloads
