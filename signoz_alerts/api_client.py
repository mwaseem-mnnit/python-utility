import logging
import os
import time
from typing import Any, Dict, List

import requests
from requests import Response
from requests.exceptions import RequestException

from config import (
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    SIGNOZ_API_KEY,
    SIGNOZ_BASE_URL,
)
from utils import payload_fingerprint


logger = logging.getLogger(__name__)


def _mock_response(payload: Dict) -> Response:
    response = Response()
    response.status_code = 200
    response._content = b'{"data":{"mock":true}}'
    response.headers["Content-Type"] = "application/json"
    logger.info("Using mock response for payload fingerprint=%s", payload_fingerprint(payload)[:12])
    return response


def _headers() -> Dict[str, str]:
    if not SIGNOZ_API_KEY:
        return {"Content-Type": "application/json"}
    return {
        "Authorization": f"Bearer {SIGNOZ_API_KEY}",
        "SIGNOZ-API-KEY": SIGNOZ_API_KEY,
        "Content-Type": "application/json",
    }


def _endpoint() -> str:
    if not SIGNOZ_BASE_URL:
        return ""
    return f"{SIGNOZ_BASE_URL}/api/v1/rules"


def _rule_url(rule_id: str) -> str:
    base = _endpoint()
    if not base or not rule_id:
        return ""
    return f"{base}/{rule_id}"


def _request(payload: Dict) -> Response:
    if os.getenv("isMock", "").lower() == "true":
        return _mock_response(payload)
    return requests.post(
        _endpoint(),
        headers=_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def fetch_existing_alert_names() -> List[Dict[str, Any]]:
    """
    Returns a list of dicts with SigNoz rule id and alert name: [{"id": "...", "alert": "..."}, ...].
    """
    endpoint = _endpoint()
    existing_alert_list: List[Dict[str, Any]] = []
    if not endpoint:
        logger.error("SIGNOZ_BASE_URL is not set. Skipping fetch of existing alerts.")
        return existing_alert_list
    if not SIGNOZ_API_KEY:
        logger.error("SIGNOZ_API_KEY is not set. Skipping fetch of existing alerts.")
        return existing_alert_list

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = requests.get(
                endpoint,
                headers=_headers(),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if 200 <= response.status_code < 300:
                payload = response.json()
                rules = payload.get("data", {}).get("rules", [])
                if isinstance(rules, list):
                    for rule in rules:
                        if not isinstance(rule, dict):
                            continue
                        name = str(rule.get("alert", "")).strip()
                        rule_id = rule.get("id")
                        if rule_id is None:
                            rule_id = rule.get("ruleId")
                        if name:
                            existing_alert_list.append(
                                {
                                    "id": "" if rule_id is None else str(rule_id),
                                    "alert": name,
                                }
                            )
                logger.info("Fetched existing alerts count=%s", len(existing_alert_list))
                return existing_alert_list

            logger.error(
                "Fetch existing alerts failed attempt=%s status=%s body=%s",
                attempt,
                response.status_code,
                response.text[:400],
            )
        except (RequestException, ValueError) as exc:
            logger.error("Fetch existing alerts error attempt=%s error=%s", attempt, exc)

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return existing_alert_list


def _delete_rule_request(rule_id: str) -> Response:
    if os.getenv("isMock", "").lower() == "true":
        response = Response()
        response.status_code = 200
        response._content = b'{"data":{"mock":true}}'
        response.headers["Content-Type"] = "application/json"
        logger.info("Using mock DELETE for rule_id=%s", rule_id)
        return response
    return requests.delete(
        _rule_url(rule_id),
        headers=_headers(),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def delete_alert(rule_id: str, rule_name: str) -> bool:
    url = _rule_url(rule_id)
    if not url:
        logger.error("Cannot delete: missing base URL or rule id.")
        return False
    if not SIGNOZ_API_KEY:
        logger.error("SIGNOZ_API_KEY is not set. Skipping delete.")
        return False

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = _delete_rule_request(rule_id)
            if 200 <= response.status_code < 300:
                logger.info("Deleted rule id=%s name=%s status=%s", rule_id, rule_name, response.status_code)
                return True
            logger.error(
                "Delete rule failed attempt=%s id=%s status=%s body=%s",
                attempt,
                rule_id,
                response.status_code,
                response.text[:400],
            )
        except RequestException as exc:
            logger.error("Delete rule request error attempt=%s id=%s error=%s", attempt, rule_id, exc)
        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    return False


def skip_alert(alert_name: str, existing_alert_list: List[Dict[str, Any]]) -> bool:
    """
    Returns True if creation should be skipped (alert already exists and delete-before-create is off).
    Returns False if caller should proceed to create (new alert, or deleted existing).
    Mutates existing_alert_list when a rule is deleted.
    """
    if not alert_name:
        return False

    delete_before = os.getenv("SIGNOZ_DELETE_BEFORE_CREATE", "false").lower() == "true"
    idx = next(
        (i for i, x in enumerate(existing_alert_list) if x.get("alert") == alert_name),
        None,
    )
    if idx is None:
        return False

    if not delete_before:
        return True

    entry = existing_alert_list[idx]
    rule_id = str(entry.get("id", "")).strip()
    rule_name = str(entry.get("alert", "")).strip()
    if not rule_id:
        logger.error(
            "Cannot delete existing alert name=%s (missing rule id); skipping create",
            alert_name,
        )
        return True

    if delete_alert(rule_id, rule_name):
        existing_alert_list.pop(idx)
    else:
        logger.error("Failed to delete existing alert name=%s id=%s; will still attempt create", alert_name, rule_id)
    return False


def create_alert(payload: Dict) -> bool:
    endpoint = _endpoint()
    fingerprint = payload_fingerprint(payload)[:12]

    if not endpoint:
        logger.error(
            "SIGNOZ_BASE_URL is not set. Skipping create_alert for payload=%s",
            fingerprint,
        )
        return False
    if not SIGNOZ_API_KEY:
        logger.error(
            "SIGNOZ_API_KEY is not set. Skipping create_alert for payload=%s",
            fingerprint,
        )
        return False

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = _request(payload)
            if 200 <= response.status_code < 300:
                logger.info(
                    "Alert created successfully status=%s alert=%s",
                    response.status_code,
                    payload.get('alert'),
                )
                return True

            logger.error(
                "Alert create failed attempt=%s status=%s payload=%s body=%s",
                attempt,
                response.status_code,
                fingerprint,
                response.text[:400],
            )
        except RequestException as exc:
            logger.error(
                "Alert create request error attempt=%s payload=%s error=%s",
                attempt,
                fingerprint,
                exc,
            )

        if attempt <= MAX_RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return False
