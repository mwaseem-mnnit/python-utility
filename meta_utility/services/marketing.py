"""Marketing orchestration service for WhatsApp template campaigns."""

from __future__ import annotations

import logging
from typing import Any

from meta_utility.core.config import MetaConfig
from meta_utility.io.csv_records import read_csv_records
from meta_utility.services.whatsapp_message import WhatsAppMessageService

logger = logging.getLogger(__name__)


class MarketingService:
    """Read marketing CSV data and send WhatsApp template messages in batches."""

    def __init__(self, whatsapp_service: WhatsAppMessageService, config: MetaConfig) -> None:
        self.whatsapp_service = whatsapp_service
        self.config = config

    def run_campaign(self) -> dict[str, Any]:
        records = read_csv_records(
            self.config.marketing_input_csv,
            delimiter=self.config.marketing_csv_delimiter,
        )
        logger.info(
            "Loaded marketing CSV rows=%s path=%s",
            len(records),
            self.config.marketing_input_csv,
        )

        messages = self._build_messages(records)
        logger.info("Prepared WhatsApp messages count=%s", len(messages))

        sent_total = 0
        failed_total = 0
        batches = 0

        for batch in self._iter_batches(messages, self.config.marketing_batch_size):
            batches += 1
            try:
                summary = self.whatsapp_service.send_template_messages(batch)
            except (ValueError, TypeError) as exc:
                logger.error("Batch %s failed before API call: %s", batches, exc)
                failed_total += len(batch)
                continue

            sent_total += int(summary["sent"])
            failed_total += int(summary["failed"])
            logger.info(
                "Batch %s completed total=%s sent=%s failed=%s",
                batches,
                summary["total"],
                summary["sent"],
                summary["failed"],
            )

        campaign_summary = {
            "rows": len(records),
            "messages": len(messages),
            "batches": batches,
            "sent": sent_total,
            "failed": failed_total,
        }
        logger.info("Marketing campaign finished: %s", campaign_summary)
        return campaign_summary

    def _build_messages(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            try:
                recipient = self._required_value(record, "phoneNumber", row_index=index)
                name = self._required_value(record, "name", row_index=index)
                discount = self._optional_with_default(
                    record,
                    "discountAmount",
                    self.config.marketing_default_discount_amount,
                )
                min_applicable = self._optional_with_default(
                    record,
                    "minApplicableAmount",
                    self.config.marketing_default_min_applicable_amount,
                )
                coupon = self._optional_with_default(
                    record,
                    "couponCode",
                    self.config.marketing_default_coupon_code,
                )
                end_date = self._optional_with_default(
                    record,
                    "endDate",
                    self.config.marketing_default_end_date,
                )
                user_limit = self._optional_with_default(
                    record,
                    "userLimit",
                    self.config.marketing_default_user_limit,
                )
            except ValueError as exc:
                logger.error("Skipping CSV row %s: %s", index, exc)
                continue

            row_values = {
                "name": name,
                "discountAmount": discount,
                "minApplicableAmount": min_applicable,
                "couponCode": coupon,
                "endDate": end_date,
                "userLimit": user_limit
            }
            parameters = [
                {"key": key, "value": row_values.get(key, "")}
                for key in self.config.marketing_template_parameter_keys
            ]
            messages.append({"recipient": recipient, "parameters": parameters})
        return messages

    @staticmethod
    def _iter_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
        batches: list[list[dict[str, Any]]] = []
        for start in range(0, len(items), batch_size):
            batches.append(items[start : start + batch_size])
        return batches

    @staticmethod
    def _required_value(record: dict[str, Any], key: str, *, row_index: int) -> str:
        value = str(record.get(key, "")).strip()
        if not value:
            raise ValueError(f"CSV row {row_index}: '{key}' is mandatory")
        return value

    @staticmethod
    def _optional_with_default(record: dict[str, Any], key: str, default: str) -> str:
        value = str(record.get(key, "")).strip()
        if value:
            return value
        return default
