"""Create missing Wix collections from a CSV source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wix_utility.catalog.matching import MatchResult, compare_catalog_objects
from wix_utility.catalog.models import CollectionDraft
from wix_utility.clients.wix_api import WixApiClient, WixApiError
from wix_utility.core.config import JOB_CREATE_COLLECTIONS, WixConfig
from wix_utility.core.utils import slugify
from wix_utility.io.csv_records import parse_csv_records
from wix_utility.jobs.base import WixJob
from wix_utility.services.collections import CollectionService

TITLE_COLUMN_CANDIDATES = (
    "collection",
    "collection_name",
    "collection title",
    "collection_title",
    "category",
    "category_name",
    "category title",
    "category_title",
    "title",
    "name",
)


@dataclass(frozen=True)
class CollectionSourceRow:
    row_number: int
    name: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class CollectionSkip:
    source: CollectionSourceRow
    matched_name: str
    result: MatchResult
    reason: str


@dataclass(frozen=True)
class CollectionCreatePlan:
    source: CollectionSourceRow
    slug: str


class CreateCollectionsJob(WixJob):
    """Create Wix collections that do not already match existing collections."""

    job_name = JOB_CREATE_COLLECTIONS

    def run(self, config: WixConfig) -> int:
        sources = self._load_sources(config)
        if sources is None:
            return 1
        if config.dry_run:
            self.logger.error(
                "WIX_DRY_RUN must be false for %s because duplicate detection requires live Wix reads. "
                "Keep WIX_COLLECTION_CREATE_ENABLED=false to plan without creating.",
                self.job_name,
            )
            return 1

        service = CollectionService(WixApiClient(config))
        try:
            existing = service.list_collections(
                page_size=max(1, config.collection_page_size),
                visible_only=config.collection_query_visible_only,
            )
        except WixApiError as exc:
            self.logger.error("Failed to fetch existing Wix collections: %s", exc)
            return 1

        self.logger.info("Loaded source collections=%s existing_wix_collections=%s", len(sources), len(existing))

        create_plans, skips = self._plan_collections(sources, existing, config.match_threshold)
        self._print_plan(create_plans, skips)

        if not config.collection_create_enabled:
            self.logger.info("Creation disabled. Set WIX_COLLECTION_CREATE_ENABLED=true to create planned collections.")
            return 0

        return self._create_planned_collections(service, create_plans, config)

    def _load_sources(self, config: WixConfig) -> list[CollectionSourceRow] | None:
        if config.input_csv is None:
            self.logger.error("WIX_INPUT_CSV is not set.")
            return None
        if not config.input_csv.is_file():
            self.logger.error("CSV file does not exist: %s", config.input_csv)
            return None

        records = parse_csv_records(config.input_csv, delimiter=config.csv_delimiter)
        sources: list[CollectionSourceRow] = []
        skipped = 0
        for index, record in enumerate(records, start=2):
            name = _collection_name_from_record(record, config.collection_title_column)
            if not name:
                skipped += 1
                self.logger.warning("Skipping CSV row %s: no collection title/name found.", index)
                continue
            sources.append(CollectionSourceRow(row_number=index, name=name, raw=record))

        self.logger.info("Parsed CSV collection rows=%s skipped_blank=%s", len(sources), skipped)
        return sources

    def _plan_collections(
        self,
        sources: list[CollectionSourceRow],
        existing: list[dict[str, Any]],
        threshold: float,
    ) -> tuple[list[CollectionCreatePlan], list[CollectionSkip]]:
        create_plans: list[CollectionCreatePlan] = []
        skips: list[CollectionSkip] = []
        planned_objects: list[dict[str, str]] = []

        for source in sources:
            existing_match = _best_match(source.name, existing, threshold)
            if existing_match is not None:
                matched_name, result = existing_match
                skips.append(
                    CollectionSkip(
                        source=source,
                        matched_name=matched_name,
                        result=result,
                        reason="already exists in Wix",
                    )
                )
                continue

            planned_match = _best_match(source.name, planned_objects, threshold)
            if planned_match is not None:
                matched_name, result = planned_match
                skips.append(
                    CollectionSkip(
                        source=source,
                        matched_name=matched_name,
                        result=result,
                        reason="duplicate in CSV/import plan",
                    )
                )
                continue

            slug = slugify(source.name)
            create_plans.append(CollectionCreatePlan(source=source, slug=slug))
            planned_objects.append({"name": source.name})

        return create_plans, skips

    def _print_plan(self, create_plans: list[CollectionCreatePlan], skips: list[CollectionSkip]) -> None:
        self.logger.info("Collections to create: %s", len(create_plans))
        for plan in create_plans:
            self.logger.info(
                "CREATE row=%s name=%r slug=%r",
                plan.source.row_number,
                plan.source.name,
                plan.slug,
            )

        self.logger.info("Collections to skip: %s", len(skips))
        for skip in skips:
            self.logger.info(
                "SKIP row=%s name=%r matched=%r score=%.4f reason=%s field_scores=%s",
                skip.source.row_number,
                skip.source.name,
                skip.matched_name,
                skip.result.score,
                skip.reason,
                skip.result.field_scores,
            )

    def _create_planned_collections(
        self,
        service: CollectionService,
        create_plans: list[CollectionCreatePlan],
        config: WixConfig,
    ) -> int:
        created = 0
        failed = 0
        for plan in create_plans:
            draft = CollectionDraft(
                name=plan.source.name,
                slug=plan.slug,
                visible=config.collection_visible,
            )
            try:
                service.create_collection(draft)
                created += 1
                self.logger.info("Created collection name=%r slug=%r", draft.name, draft.slug)
            except WixApiError as exc:
                failed += 1
                self.logger.error("Failed to create collection name=%r error=%s", draft.name, exc)

        self.logger.info("Create collections complete: created=%s failed=%s", created, failed)
        return 1 if failed else 0


def _collection_name_from_record(record: dict[str, Any], configured_column: str) -> str:
    if configured_column:
        return str(_get_case_insensitive(record, configured_column) or "").strip()
    for candidate in TITLE_COLUMN_CANDIDATES:
        value = _get_case_insensitive(record, candidate)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _get_case_insensitive(record: dict[str, Any], key: str) -> Any:
    normalized_key = _normalize_key(key)
    for raw_key, value in record.items():
        if _normalize_key(raw_key) == normalized_key:
            return value
    return None


def _normalize_key(value: str) -> str:
    return value.strip().casefold().replace(" ", "_").replace("-", "_")


def _best_match(
    source_name: str,
    candidates: list[dict[str, Any]],
    threshold: float,
) -> tuple[str, MatchResult] | None:
    best_name = ""
    best_result: MatchResult | None = None
    source_obj = {"name": source_name}

    for candidate in candidates:
        candidate_name = _collection_name_from_existing(candidate)
        if not candidate_name:
            continue
        candidate_obj = {
            "name": candidate_name,
            "description": _collection_description_from_existing(candidate),
        }
        result = compare_catalog_objects(source_obj, candidate_obj, threshold=threshold)
        if best_result is None or result.score > best_result.score:
            best_result = result
            best_name = candidate_name

    if best_result is None or not best_result.is_match:
        return None
    return best_name, best_result


def _collection_name_from_existing(collection: dict[str, Any]) -> str:
    nested = collection.get("collection")
    nested_collection = nested if isinstance(nested, dict) else {}
    return str(
        collection.get("name")
        or collection.get("title")
        or nested_collection.get("name")
        or nested_collection.get("title")
        or ""
    ).strip()


def _collection_description_from_existing(collection: dict[str, Any]) -> str:
    nested = collection.get("collection")
    nested_collection = nested if isinstance(nested, dict) else {}
    return str(
        collection.get("description")
        or nested_collection.get("description")
        or ""
    ).strip()
