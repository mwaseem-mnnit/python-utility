"""WebP conversion helpers and compress CLI entry (orchestration lives in ``pipeline.runner``)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PIL import Image

LOGGER = logging.getLogger(__name__)


def convert_to_webp(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    width: int,
    height: int,
) -> None:
    """Resize an image to ``(width, height)`` and save as WebP."""
    with Image.open(input_path) as img:
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "webp", quality=100)


def compute_product_info_images(
    input_dir: os.PathLike[str],
    output_dir: os.PathLike[str],
    *,
    is_thumbnail: bool = False,
    logger: logging.Logger | None = None,
    max_files: int | None = None,
) -> tuple[int, int]:
    """
    Convert JPG/JPEG/PNG files in ``input_dir`` to WebP under ``output_dir``.

    Uses the centralized pipeline with a single ``compress`` step so behavior matches
    :func:`run`. Kept for callers that invoke the utility programmatically.
    """
    from image_utility.pipeline.runner import run_pipeline

    log = logger or LOGGER
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")

    if is_thumbnail:
        os.environ["IMAGE_UTIL_THUMBNAIL"] = "1"
    else:
        os.environ.pop("IMAGE_UTIL_THUMBNAIL", None)

    prev_in = os.environ.get("IMAGE_UTIL_INPUT_DIR")
    prev_out = os.environ.get("IMAGE_UTIL_OUTPUT_DIR")
    prev_max = os.environ.get("IMAGE_UTIL_MAX_FILES")
    prev_steps = os.environ.get("IMAGE_UTIL_PIPELINE_STEPS")

    try:
        os.environ["IMAGE_UTIL_INPUT_DIR"] = str(input_path.resolve())
        os.environ["IMAGE_UTIL_OUTPUT_DIR"] = str(output_path.resolve())
        if max_files is not None:
            os.environ["IMAGE_UTIL_MAX_FILES"] = str(max_files)
        else:
            os.environ.pop("IMAGE_UTIL_MAX_FILES", None)
        os.environ["IMAGE_UTIL_PIPELINE_STEPS"] = "compress"
        summary = run_pipeline(steps=["compress"])
    finally:
        _restore_env("IMAGE_UTIL_INPUT_DIR", prev_in)
        _restore_env("IMAGE_UTIL_OUTPUT_DIR", prev_out)
        _restore_env("IMAGE_UTIL_MAX_FILES", prev_max)
        _restore_env("IMAGE_UTIL_PIPELINE_STEPS", prev_steps)

    if summary.exit_code != 0:
        log.error("Pipeline compress job exited with code %s.", summary.exit_code)
        return 0, 0

    return summary.processed, summary.skipped


def _restore_env(key: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def run() -> int:
    """Run WebP conversion via the shared pipeline orchestrator."""
    from image_utility.pipeline.runner import run_pipeline
    from image_utility.utils import init_job_logging, load_image_utility_env

    load_image_utility_env()
    init_job_logging("compress.log")
    return run_pipeline(steps=["compress"]).exit_code
