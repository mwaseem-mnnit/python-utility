"""Import product titles/images from a WhatsApp export folder and generate a CSV."""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app_logging import init_logging

ATTACHMENT_RE = re.compile(r"<attached:\s*([^>]+)>", re.IGNORECASE)
MESSAGE_RE = re.compile(r"^\[[^]]+\]\s[^:]+:\s?(.*)$", re.DOTALL)
MESSAGE_START_RE = re.compile(r"^\[[^]]+\]\s")

CONTROL_CHARS = {
    "\ufeff",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2060",
}


@dataclass(frozen=True)
class Settings:
    input_dir: Path
    output_dir: Path
    product_start_id: int
    chat_file: str = "_chat.txt"
    output_csv: str = "products.csv"
    log_file: Path | None = None


@dataclass(frozen=True)
class ProductDraft:
    attachments: list[str]
    title: str


def _clean_text(value: str) -> str:
    return "".join(ch for ch in value if ch not in CONTROL_CHARS).strip()


def _iter_messages(chat_text: str) -> list[str]:
    messages: list[str] = []
    current: list[str] = []

    for line in chat_text.splitlines():
        normalized = _clean_text(line)
        if MESSAGE_START_RE.match(normalized):
            if current:
                messages.append("\n".join(current).strip())
            current = [normalized]
            continue
        if current:
            current.append(normalized)

    if current:
        messages.append("\n".join(current).strip())
    return messages


def _extract_body(raw_message: str) -> str:
    normalized = _clean_text(raw_message)
    match = MESSAGE_RE.match(normalized)
    return _clean_text(match.group(1) if match else normalized)


def parse_products(chat_file: Path) -> list[ProductDraft]:
    text = chat_file.read_text(encoding="utf-8", errors="ignore")
    messages = _iter_messages(text)

    drafts: list[ProductDraft] = []
    pending_attachments: list[str] = []
    pending_title_lines: list[str] = []

    for message in messages:
        body = _extract_body(message)
        attachment_match = ATTACHMENT_RE.search(body)

        if attachment_match:
            if pending_attachments and pending_title_lines:
                drafts.append(
                    ProductDraft(
                        attachments=list(pending_attachments),
                        title=" ".join(pending_title_lines).strip(),
                    )
                )
                pending_attachments.clear()
                pending_title_lines.clear()

            pending_attachments.append(Path(attachment_match.group(1).strip()).name)
            continue

        if not pending_attachments:
            continue

        if not body or body.startswith("<"):
            continue

        pending_title_lines.append(" ".join(part for part in body.splitlines() if part.strip()))

    if pending_attachments:
        title = " ".join(pending_title_lines).strip() or "Untitled"
        drafts.append(ProductDraft(attachments=list(pending_attachments), title=title))

    return drafts


def _resolve_file(input_dir: Path, file_name: str) -> Path | None:
    direct = input_dir / file_name
    if direct.exists():
        return direct

    lower = file_name.lower()
    for path in input_dir.iterdir():
        if path.is_file() and path.name.lower() == lower:
            return path
    return None


def process_whatsapp_export(
    *,
    input_dir: Path,
    output_dir: Path,
    product_start_id: int,
    chat_filename: str,
    output_csv_name: str,
    logger: logging.Logger,
) -> tuple[int, int]:
    chat_path = input_dir / chat_filename
    if not chat_path.exists():
        raise FileNotFoundError(f"Chat file not found: {chat_path}")

    drafts = parse_products(chat_path)
    logger.info("Discovered %s product groups from %s", len(drafts), chat_path.name)

    rows: list[dict[str, str]] = []
    processed_products = 0

    for offset, draft in enumerate(drafts):
        identifier = product_start_id + offset
        renamed_images: list[str] = []

        for image_index, original_name in enumerate(draft.attachments, start=1):
            source_path = _resolve_file(input_dir, original_name)
            if source_path is None:
                logger.warning(
                    "Missing attachment for identifier=%s: %s", identifier, original_name
                )
                continue

            extension = source_path.suffix.lower() or ".jpg"
            target_name = f"{identifier}_{image_index}{extension}"
            target_path = input_dir / target_name

            if source_path.resolve() != target_path.resolve():
                if target_path.exists():
                    raise FileExistsError(
                        f"Cannot rename {source_path.name}; target exists: {target_path.name}"
                    )
                source_path.rename(target_path)

            renamed_images.append(target_name)

        if not renamed_images:
            logger.warning("Skipping identifier=%s because no images were renamed", identifier)
            continue

        rows.append(
            {
                "identifier": str(identifier),
                "title": draft.title,
                "image_names": ",".join(renamed_images),
            }
        )
        processed_products += 1

    output_csv = output_dir / output_csv_name
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identifier", "title", "image_names"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote CSV: %s (%s rows)", output_csv, len(rows))
    return processed_products, len(rows)


def _load_settings() -> Settings:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    load_dotenv(script_dir / ".env")

    input_raw = os.getenv("WHATSAPP_INPUT_DIR", "").strip()
    if not input_raw:
        raise ValueError("WHATSAPP_INPUT_DIR is required in scripts/.env")

    output_raw = os.getenv("WHATSAPP_OUTPUT_DIR", "").strip()
    if not output_raw:
        raise ValueError("WHATSAPP_OUTPUT_DIR is required in scripts/.env")


    product_raw = os.getenv("WHATSAPP_PRODUCT_START_ID", "").strip()
    if not product_raw or not product_raw.isdigit():
        raise ValueError("WHATSAPP_PRODUCT_START_ID must be a positive integer")

    log_raw = os.getenv("WHATSAPP_LOG_FILE", "log/whatsapp_products.log").strip()
    log_path = Path(log_raw).expanduser()
    if not log_path.is_absolute():
        log_path = (workspace_root / log_path).resolve()

    return Settings(
        input_dir=Path(input_raw).expanduser().resolve(),
        output_dir=Path(output_raw).expanduser().resolve(),
        product_start_id=int(product_raw, 10),
        chat_file=os.getenv("WHATSAPP_CHAT_FILE", "_chat.txt").strip() or "_chat.txt",
        output_csv=os.getenv("WHATSAPP_OUTPUT_CSV", "products.csv").strip() or "products.csv",
        log_file=log_path,
    )


def main() -> int:
    try:
        settings = _load_settings()
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return 1

    if not settings.input_dir.exists() or not settings.input_dir.is_dir():
        print(f"Input directory not found: {settings.input_dir}")
        return 1

    init_logging(log_file=settings.log_file, also_stdout=True)
    logger = logging.getLogger(__name__)

    try:
        processed, row_count = process_whatsapp_export(
            input_dir=settings.input_dir,
            output_dir=settings.output_dir,
            product_start_id=settings.product_start_id,
            chat_filename=settings.chat_file,
            output_csv_name=settings.output_csv,
            logger=logger,
        )
    except Exception as exc:
        logger.exception("Import failed: %s", exc)
        return 1

    logger.info(
        "Completed WhatsApp import. products=%s csv_rows=%s output=%s",
        processed,
        row_count,
        settings.input_dir / settings.output_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


