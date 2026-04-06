import os
from copy import deepcopy
from typing import Any, Dict

from utils import load_env_file


load_env_file()


SIGNOZ_BASE_URL = os.getenv("SIGNOZ_BASE_URL", "").rstrip("/")
SIGNOZ_API_KEY = os.getenv("SIGNOZ_API_KEY", "")
CSV_PATH = os.getenv("SIGNOZ_ALERTS_CSV_PATH", "")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("SIGNOZ_REQUEST_TIMEOUT_SECONDS", "15"))
MAX_RETRIES = int(os.getenv("SIGNOZ_MAX_RETRIES", "2"))
RETRY_BACKOFF_SECONDS = float(os.getenv("SIGNOZ_RETRY_BACKOFF_SECONDS", "1.5"))
DEFAULT_PREFERRED_CHANNEL = os.getenv("SIGNOZ_PREFERRED_CHANNEL", "")
SIGNOZ_POD_LABEL = os.getenv("SIGNOZ_POD_LABEL", "wallet")

DEFAULT_P99_THRESHOLD_MS = float(os.getenv("SIGNOZ_P99_THRESHOLD_MS", "250"))
DEFAULT_P50_THRESHOLD_MS = float(os.getenv("SIGNOZ_P50_THRESHOLD_MS", "120"))
DEFAULT_ERROR_RATE_THRESHOLD = float(os.getenv("SIGNOZ_ERROR_RATE_THRESHOLD", "0.1"))
LATENCY_THRESHOLD_INCREASE_PERCENT = float(
    os.getenv("SIGNOZ_LATENCY_THRESHOLD_INCREASE_PERCENT", "20")
)
ERROR_RATE_THRESHOLD_INCREASE_PERCENT = float(
    os.getenv("SIGNOZ_ERROR_RATE_THRESHOLD_INCREASE_PERCENT", "20")
)
DEFAULT_CPU_THRESHOLD = float(os.getenv("SIGNOZ_CPU_THRESHOLD", "0.75"))
DEFAULT_MEMORY_THRESHOLD_MBY = float(os.getenv("SIGNOZ_MEMORY_THRESHOLD_MBY", "350"))

ALERT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "latency": {
        "state": "active",
        "alert": "{service_name}-{route_slug}-{metric_key}-latency-high",
        "alertType": "METRIC_BASED_ALERT",
        "ruleType": "threshold_rule",
        "evalWindow": "5m",
        "frequency": "1m",
        "condition": {
            "compositeQuery": {
                "queries": [
                    {
                        "type": "builder_query",
                        "spec": {
                            "name": "A",
                            "stepInterval": 0,
                            "signal": "metrics",
                            "source": "",
                            "aggregations": [
                                {
                                    "metricName": "http.server.duration.bucket",
                                    "temporality": "",
                                    "timeAggregation": "rate",
                                    "spaceAggregation": "{space_aggregation}",
                                    "reduceTo": "",
                                }
                            ],
                            "filter": {
                                "expression": (
                                    "k8s.deployment.name = '{service_name}' and "
                                    "http.route = '{route_name}' and http.route EXISTS AND http.method EXISTS"
                                )
                            },
                            "having": {"expression": ""},
                        },
                    }
                ],
                "panelType": "graph",
                "queryType": "builder",
                "unit": "ms",
            },
            "selectedQueryName": "A",
            "thresholds": {
                "kind": "basic",
                "spec": [
                    {
                        "name": "critical",
                        "target": "{threshold}",
                        "targetUnit": "ms",
                        "recoveryTarget": None,
                        "matchType": "2",
                        "op": "1",
                        "channels": ["{SIGNOZ_PREFERRED_CHANNEL}"],
                    }
                ],
            },
        },
        "labels": {"pod": "{SIGNOZ_POD_LABEL}", "service": "{service_name}"},
        "annotations": {
            "description": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
            "summary": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
        },
        "disabled": False,
        "source": "{SIGNOZ_BASE_URL}/alerts/new?ruleType=threshold_rule&alertType=METRIC_BASED_ALERT",
        "version": "v5",
        "evaluation": {"kind": "rolling", "spec": {"evalWindow": "5m0s", "frequency": "1m"}},
        "schemaVersion": "v2alpha1",
        "notificationSettings": {"renotify": {"enabled": False, "interval": "30m"}},
    },
    "error_rate": {
        "state": "active",
        "alert": "{service_name}-{route_slug}-{metric_key}-high",
        "alertType": "METRIC_BASED_ALERT",
        "ruleType": "threshold_rule",
        "evalWindow": "5m",
        "frequency": "1m",
        "condition": {
            "compositeQuery": {
                "queries": [
                    {
                        "type": "builder_query",
                        "spec": {
                            "name": "A",
                            "stepInterval": 0,
                            "signal": "metrics",
                            "source": "",
                            "aggregations": [
                                {
                                    "metricName": "http.server.duration.count",
                                    "temporality": "",
                                    "timeAggregation": "rate",
                                    "spaceAggregation": "sum",
                                    "reduceTo": "",
                                }
                            ],
                            "disabled": True,
                            "filter": {
                                "expression": (
                                    "k8s.deployment.name = '{service_name}' AND "
                                    "http.route = '{route_name}' "
                                    "and http.status_code NOT ilike '2%' and http.status_code NOT ilike '3%' "
                                )
                            },
                            "having": {"expression": ""},
                        },
                    },
                    {
                        "type": "builder_query",
                        "spec": {
                            "name": "B",
                            "stepInterval": 0,
                            "signal": "metrics",
                            "source": "",
                            "aggregations": [
                                {
                                    "metricName": "http.server.duration.count",
                                    "temporality": "",
                                    "timeAggregation": "rate",
                                    "spaceAggregation": "sum",
                                    "reduceTo": "",
                                }
                            ],
                            "disabled": True,
                            "filter": {
                                "expression": (
                                    "k8s.deployment.name = '{service_name}' AND "
                                    "http.route = '{route_name}' "
                                )
                            },
                            "having": {"expression": ""},
                        },
                    },
                    {"type": "builder_formula", "spec": {"name": "F1", "expression": "(A/B)"}},
                ],
                "panelType": "graph",
                "queryType": "builder",
            },
            "selectedQueryName": "F1",
            "thresholds": {
                "kind": "basic",
                "spec": [
                    {
                        "name": "critical",
                        "target": "{threshold}",
                        "targetUnit": "",
                        "recoveryTarget": None,
                        "matchType": "2",
                        "op": "1",
                        "channels": ["{SIGNOZ_PREFERRED_CHANNEL}"],
                    }
                ],
            },
        },
        "labels": {"pod": "{SIGNOZ_POD_LABEL}", "service": "{service_name}"},
        "annotations": {
            "description": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
            "summary": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
        },
        "disabled": False,
        "source": "{SIGNOZ_BASE_URL}/alerts/new?ruleType=threshold_rule&alertType=METRIC_BASED_ALERT",
        "version": "v5",
        "evaluation": {"kind": "rolling", "spec": {"evalWindow": "5m0s", "frequency": "1m"}},
        "schemaVersion": "v2alpha1",
        "notificationSettings": {"renotify": {"enabled": False, "interval": "30m"}},
    },
    "cpu_high": {
        "state": "active",
        "alert": "{service_name}-{metric_key}-high",
        "alertType": "METRIC_BASED_ALERT",
        "ruleType": "threshold_rule",
        "evalWindow": "5m",
        "frequency": "1m",
        "condition": {
            "compositeQuery": {
                "queries": [
                    {
                        "type": "builder_query",
                        "spec": {
                            "name": "A",
                            "stepInterval": 0,
                            "signal": "metrics",
                            "source": "",
                            "aggregations": [
                                {
                                    "metricName": "k8s.pod.cpu.usage",
                                    "temporality": "",
                                    "timeAggregation": "avg",
                                    "spaceAggregation": "avg",
                                    "reduceTo": "",
                                }
                            ],
                            "filter": {"expression": "k8s.deployment.name = '{service_name}'"},
                            "having": {"expression": ""},
                        },
                    }
                ],
                "panelType": "graph",
                "queryType": "builder",
                "unit": "percentunit",
            },
            "selectedQueryName": "A",
            "thresholds": {
                "kind": "basic",
                "spec": [
                    {
                        "name": "critical",
                        "target": "{threshold}",
                        "targetUnit": "percentunit",
                        "recoveryTarget": None,
                        "matchType": "4",
                        "op": "1",
                        "channels": ["{SIGNOZ_PREFERRED_CHANNEL}"],
                    }
                ],
            },
        },
        "labels": {"pod": "{SIGNOZ_POD_LABEL}", "service": "{service_name}"},
        "annotations": {
            "description": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
            "summary": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
        },
        "disabled": False,
        "source": "{SIGNOZ_BASE_URL}/alerts/new?ruleType=threshold_rule&alertType=METRIC_BASED_ALERT",
        "version": "v5",
        "evaluation": {"kind": "rolling", "spec": {"evalWindow": "5m0s", "frequency": "1m"}},
        "schemaVersion": "v2alpha1",
        "notificationSettings": {"renotify": {"enabled": False, "interval": "30m"}},
    },
    "memory_high": {
        "state": "active",
        "alert": "{service_name}-{metric_key}-high",
        "alertType": "METRIC_BASED_ALERT",
        "ruleType": "threshold_rule",
        "evalWindow": "5m",
        "frequency": "1m",
        "condition": {
            "compositeQuery": {
                "queries": [
                    {
                        "type": "builder_query",
                        "spec": {
                            "name": "A",
                            "stepInterval": 0,
                            "signal": "metrics",
                            "source": "",
                            "aggregations": [
                                {
                                    "metricName": "k8s.pod.memory.usage",
                                    "temporality": "",
                                    "timeAggregation": "avg",
                                    "spaceAggregation": "avg",
                                    "reduceTo": "",
                                }
                            ],
                            "filter": {"expression": "k8s.deployment.name = '{service_name}'"},
                            "having": {"expression": ""},
                        },
                    }
                ],
                "panelType": "graph",
                "queryType": "builder",
                "unit": "By",
            },
            "selectedQueryName": "A",
            "thresholds": {
                "kind": "basic",
                "spec": [
                    {
                        "name": "critical",
                        "target": "{threshold}",
                        "targetUnit": "MBy",
                        "recoveryTarget": None,
                        "matchType": "2",
                        "op": "1",
                        "channels": ["{SIGNOZ_PREFERRED_CHANNEL}"],
                    }
                ],
            },
        },
        "labels": {"pod": "{SIGNOZ_POD_LABEL}", "service": "{service_name}"},
        "annotations": {
            "description": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
            "summary": "This alert is fired when the defined metric (current value: {{$value}}) crosses the threshold ({{$threshold}})",
        },
        "disabled": False,
        "source": "{SIGNOZ_BASE_URL}/alerts/new?ruleType=threshold_rule&alertType=METRIC_BASED_ALERT",
        "version": "v5",
        "evaluation": {"kind": "rolling", "spec": {"evalWindow": "5m0s", "frequency": "1m"}},
        "schemaVersion": "v2alpha1",
        "notificationSettings": {"renotify": {"enabled": False, "interval": "30m"}},
    },
}


def get_template(template_key: str) -> Dict[str, Any]:
    return deepcopy(ALERT_TEMPLATES[template_key])
