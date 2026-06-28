"""Fuzzy matching helpers to avoid duplicate catalog records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, is_dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping

TEXT_FIELDS = ("title", "name", "description")
DEFAULT_MATCH_THRESHOLD = 0.86


@dataclass(frozen=True)
class MatchResult:
    """Detailed comparison result for two product or collection-like objects."""

    is_match: bool
    score: float
    threshold: float
    field_scores: dict[str, float]
    reasons: tuple[str, ...]


def compare_catalog_objects(
    left: Any,
    right: Any,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> MatchResult:
    """Compare two objects using name/title and description similarity."""
    left_map = _object_to_mapping(left)
    right_map = _object_to_mapping(right)

    name_score = max(
        _field_similarity(left_map, right_map, "name"),
        _field_similarity(left_map, right_map, "title"),
        _cross_field_similarity(left_map, right_map, "name", "title"),
        _cross_field_similarity(left_map, right_map, "title", "name"),
    )
    description_score = _field_similarity(left_map, right_map, "description")
    token_score = _token_similarity(_best_name(left_map), _best_name(right_map))

    field_scores = {
        "name": round(name_score, 4),
        "description": round(description_score, 4),
        "name_tokens": round(token_score, 4),
    }
    left_description = str(left_map.get("description", "")).strip()
    right_description = str(right_map.get("description", "")).strip()
    if left_description or right_description:
        score = (name_score * 0.60) + (token_score * 0.30) + (description_score * 0.10)
    else:
        score = (name_score * 0.65) + (token_score * 0.35)
    reasons = _reasons(name_score, token_score, description_score)
    return MatchResult(
        is_match=score >= threshold,
        score=round(score, 4),
        threshold=threshold,
        field_scores=field_scores,
        reasons=tuple(reasons),
    )


def are_catalog_objects_same(left: Any, right: Any, *, threshold: float = DEFAULT_MATCH_THRESHOLD) -> bool:
    """Boolean convenience wrapper around :func:`compare_catalog_objects`."""
    return compare_catalog_objects(left, right, threshold=threshold).is_match


def _object_to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _best_name(value: Mapping[str, Any]) -> str:
    return str(value.get("name") or value.get("title") or "")


def _field_similarity(left: Mapping[str, Any], right: Mapping[str, Any], field: str) -> float:
    return _string_similarity(str(left.get(field, "")), str(right.get(field, "")))


def _cross_field_similarity(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_field: str,
    right_field: str,
) -> float:
    return _string_similarity(str(left.get(left_field, "")), str(right.get(right_field, "")))


def _string_similarity(left: str, right: str) -> float:
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        return max(0.9, shorter / longer)
    compact_left = left_norm.replace(" ", "")
    compact_right = right_norm.replace(" ", "")
    compact_score = SequenceMatcher(None, compact_left, compact_right).ratio()
    spaced_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    return max(compact_score, spaced_score)


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_normalize_text(left).split())
    right_tokens = set(_normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    jaccard = overlap / len(left_tokens | right_tokens)
    compact_left = _normalize_text(left).replace(" ", "")
    compact_right = _normalize_text(right).replace(" ", "")
    left_contained = all(token in compact_right for token in left_tokens)
    right_contained = all(token in compact_left for token in right_tokens)
    containment = 0.95 if left_contained and right_contained else 0.0
    return max(jaccard, containment)


def _normalize_text(value: str) -> str:
    lowered = value.casefold()
    without_punct = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", without_punct).strip()


def _reasons(name_score: float, token_score: float, description_score: float) -> list[str]:
    reasons: list[str] = []
    if name_score >= 0.95:
        reasons.append("name/title nearly identical")
    elif name_score >= 0.85:
        reasons.append("name/title strongly similar")
    if token_score >= 0.75:
        reasons.append("name/title keywords overlap")
    if description_score >= 0.85:
        reasons.append("descriptions strongly similar")
    if not reasons:
        reasons.append("no strong matching signal")
    return reasons
