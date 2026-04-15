"""Batch convert JPEG images from an input folder to WebP files."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app_logging import init_logging
from image_utility.compress.jpg_webp import convert_to_webp

ENV_INPUT_DIR = "IMAGE_UTIL_INPUT_DIR"
ENV_OUTPUT_DIR = "IMAGE_UTIL_OUTPUT_DIR"
_COMPRESS_DIR = Path(__file__).resolve().parent
JPEG_EXTS = {".jpg", ".jpeg", ".png"}
WEBP_SIZE = 950
THUMBNAIL_SIZE = 420


def _stem_trailing_index(stem: str) -> int | None:
    """Parse ``<identifier>_<index>`` from filename stem; return index or None."""
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail, 10)


def _resolve_dir_from_env(var_name: str) -> Path | None:
    raw = os.getenv(var_name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def compute_product_info_images(
    input_dir: Path,
    output_dir: Path,
    *,
    is_thumbnail: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[int, int]:
    """
    Convert JPG/JPEG/PNG files in ``input_dir`` to WebP under ``output_dir``.

    When ``is_thumbnail`` is False, every JPEG is resized to ``WEBP_SIZE`` and
    written next to ``output_dir``.

    When ``is_thumbnail`` is True, only files whose stem matches
    ``<identifier>_0`` (index ``0`` after the last underscore) are converted,
    resized to ``THUMBNAIL_SIZE``, and written under ``output_dir/thumbnail``.

    Returns ``(ok_count, skipped_count)``.
    """
    log = logger or logging.getLogger(__name__)
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    size = THUMBNAIL_SIZE if is_thumbnail else WEBP_SIZE
    dest_root = output_dir / "thumbnail" if is_thumbnail else output_dir
    dest_root.mkdir(parents=True, exist_ok=True)

    log.info("Input directory: %s", input_dir)
    log.info("Output directory: %s", dest_root)
    if is_thumbnail:
        log.info("Thumbnail mode: index 0 only, size %s", THUMBNAIL_SIZE)

    files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in JPEG_EXTS
    )
    if is_thumbnail:
        files = [p for p in files if _stem_trailing_index(p.stem) == 0]

    ok, skipped = 0, 0
    max_iteration = int(os.getenv("IMAGE_UTIL_MAX_FILES", 1))

    for source_path in files:
        if (ok + skipped) >= max_iteration:
            log.info("breaking loop on max iteration: %s", max_iteration)
            break

        destination_path = dest_root / f"{source_path.stem}.webp"
        try:
            convert_to_webp(
                source_path,
                destination_path,
                size,
                size,
            )
            ok += 1
            log.info("Converted %s -> %s", source_path.name, destination_path.name)
        except OSError as exc:
            skipped += 1
            log.warning("Skip %s: %s", source_path.name, exc)

    return ok, skipped


def main() -> int:
    load_dotenv(_COMPRESS_DIR / ".env")
    init_logging(also_stdout=True, default_filename="app.log")
    logger = logging.getLogger(__name__)

    input_dir = _resolve_dir_from_env(ENV_INPUT_DIR)
    output_dir = _resolve_dir_from_env(ENV_OUTPUT_DIR)

    if input_dir is None:
        logger.error("%s is not set in .env.", ENV_INPUT_DIR)
        return 1
    if output_dir is None:
        logger.error("%s is not set in .env.", ENV_OUTPUT_DIR)
        return 1

    try:
        ok, skipped = compute_product_info_images(input_dir, output_dir, logger=logger, is_thumbnail=False)
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. %s image(s) converted, %s skipped.", ok, skipped)
    return 0
