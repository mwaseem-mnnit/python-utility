"""Job dispatcher for meta utility commands."""

from __future__ import annotations

import logging
import sys

from meta_utility.clients.graph_facebook_api import GraphFacebookApiClient
from meta_utility.core.config import JOB_WHATSAPP_MARKETING, MetaConfig, load_meta_config
from meta_utility.core.logging import init_job_logging
from meta_utility.services.marketing import MarketingService
from meta_utility.services.whatsapp_message import WhatsAppMessageService

logger = logging.getLogger(__name__)


def run_job(job: str) -> int:
    key = job.strip().lower()
    if key != JOB_WHATSAPP_MARKETING:
        print(
            f"Unknown meta utility job '{job}'. Valid jobs: {JOB_WHATSAPP_MARKETING}",
            file=sys.stderr,
        )
        return 2

    init_job_logging("meta_utility.log")
    logger.info("Starting job: %s", JOB_WHATSAPP_MARKETING)
    try:
        config = load_meta_config()
        exit_code = _run_whatsapp_marketing(config)
    except (ValueError, OSError) as exc:
        logger.error("Configuration/runtime error: %s", exc)
        return 1
    logger.info("Finished job: %s exit_code=%s", JOB_WHATSAPP_MARKETING, exit_code)
    return exit_code


def _run_whatsapp_marketing(config: MetaConfig) -> int:
    client = GraphFacebookApiClient(config)
    whatsapp_service = WhatsAppMessageService(client, config)
    marketing_service = MarketingService(whatsapp_service, config)
    summary = marketing_service.run_campaign()
    return 0 if summary["failed"] == 0 else 1

