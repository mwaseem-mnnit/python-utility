"""WebP conversion processor."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

from image_utility.utils import (
    init_job_logging,
    load_job_env,
    parse_positive_int_env,
    resolve_dir_from_env,
    sorted_image_files,
)

ENV_INPUT_DIR = "IMAGE_UTIL_INPUT_DIR"
ENV_OUTPUT_DIR = "IMAGE_UTIL_OUTPUT_DIR"
ENV_MAX_FILES = "IMAGE_UTIL_MAX_FILES"
WEBP_SIZE = 950
THUMBNAIL_SIZE = 420
_MODULE_DIR = Path(__file__).resolve().parent


def _stem_trailing_index(stem: str) -> int | None:
    """Parse ``<identifier>_<index>`` from filename stem; return index or None."""
    if "_" not in stem:
        return None
    tail = stem.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail, 10)


def convert_to_webp(
    input_path: str | Path,
    output_path: str | Path,
    width: int,
    height: int,
) -> None:
    """Resize an image to ``(width, height)`` and save as WebP."""
    with Image.open(input_path) as img:
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "webp", quality=100)


def compute_product_info_images(
    input_dir: Path,
    output_dir: Path,
    *,
    is_thumbnail: bool = False,
    logger: logging.Logger | None = None,
    max_files: int | None = None,
) -> tuple[int, int]:
    """
    Convert JPG/JPEG/PNG files in ``input_dir`` to WebP under ``output_dir``.

    When ``is_thumbnail`` is True, only files whose stem matches ``<identifier>_0``
    are converted into ``output_dir/thumbnail``.
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

    files = sorted_image_files(input_dir)
    if is_thumbnail:
        files = [p for p in files if _stem_trailing_index(p.stem) == 0]

    ok, skipped = 0, 0
    for source_path in files:
        if max_files is not None and (ok + skipped) >= max_files:
            log.info("Reached %s=%s, stopping.", ENV_MAX_FILES, max_files)
            break

        destination_path = dest_root / f"{source_path.stem}.webp"
        try:
            convert_to_webp(source_path, destination_path, size, size)
            ok += 1
            log.info("Converted %s -> %s", source_path.name, destination_path.name)
        except OSError as exc:
            skipped += 1
            log.warning("Skip %s: %s", source_path.name, exc)

    return ok, skipped


def run() -> int:
    """Run the WebP conversion job from ``compress/.env``."""
    load_job_env(_MODULE_DIR)
    init_job_logging(default_filename="compress.log")
    logger = logging.getLogger(__name__)

    input_dir = resolve_dir_from_env(ENV_INPUT_DIR)
    output_dir = resolve_dir_from_env(ENV_OUTPUT_DIR)
    max_files = parse_positive_int_env(ENV_MAX_FILES)

    if input_dir is None:
        logger.error("%s is not set in .env.", ENV_INPUT_DIR)
        return 1
    if output_dir is None:
        logger.error("%s is not set in .env.", ENV_OUTPUT_DIR)
        return 1

    try:
        ok, skipped = compute_product_info_images(
            input_dir,
            output_dir,
            logger=logger,
            max_files=max_files,
            is_thumbnail=os.getenv("IMAGE_UTIL_THUMBNAIL", "").strip() == "1",
        )
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Done. %s image(s) converted, %s skipped.", ok, skipped)
    return 0


