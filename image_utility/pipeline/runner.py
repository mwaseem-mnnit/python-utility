"""Central orchestration: env, file iteration, phase sequence, output write."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_utility.config import WORKSPACE_ROOT, ENV_PIPELINE_STEPS, ENV_THUMBNAIL, ENV_INPUT_DIR, ENV_OUTPUT_DIR, \
    ENV_MAX_FILES
from image_utility.pipeline.context import PipelineContext
from image_utility.pipeline.registry import resolve_pipeline_phases
from image_utility.utils import (
    init_job_logging,
    load_image_utility_env,
    parse_positive_int_env,
    resolve_dir_from_env,
    sorted_image_files,
    stem_trailing_index,
)


@dataclass(frozen=True)
class PipelineRunSummary:
    exit_code: int
    processed: int
    skipped: int


def _parse_steps_from_env() -> list[str]:
    raw = os.getenv(ENV_PIPELINE_STEPS, "").strip()
    if not raw:
        return ["compress"]
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _ensure_debug_layout() -> None:
    root = WORKSPACE_ROOT / "debug"
    for name in ("isolate", "compose", "shadow", "polish"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "decomposition").mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "filtering").mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "ranking").mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "grouping").mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "ownership").mkdir(parents=True, exist_ok=True)
    (root / "isolate" / "suppression").mkdir(parents=True, exist_ok=True)


def _thumbnail_mode_active(steps: list[str]) -> bool:
    return os.getenv(ENV_THUMBNAIL, "").strip() == "1" and "compress" in {s.lower() for s in steps}


def _output_directory_for_job(base_output: Path, steps: list[str]) -> Path:
    if _thumbnail_mode_active(steps):
        return base_output / "thumbnail"
    return base_output


def _filter_thumbnail_files(files: list[Path]) -> list[Path]:
    return [p for p in files if stem_trailing_index(p.stem) == 0]


def _load_source_into_context(ctx: PipelineContext, logger: logging.Logger) -> bool:
    try:
        with Image.open(ctx.input_path) as img:
            ctx.current_image = np.array(img.convert("RGB"))
        ctx.current_rgba = None
        ctx.alpha_mask = None
        return True
    except OSError as exc:
        logger.warning("Skip %s: cannot load image (%s).", ctx.input_path.name, exc)
        return False


def _write_final_output(ctx: PipelineContext, logger: logging.Logger) -> bool:
    try:
        ctx.output_path.mkdir(parents=True, exist_ok=True)
        stem = ctx.input_path.stem
        write_format = (ctx.metadata.get("write_format") or "jpeg").lower()

        if write_format == "webp" and ctx.current_image is not None:
            dest = ctx.output_path / f"{stem}.webp"
            quality = int(ctx.metadata.get("webp_quality", 100))
            Image.fromarray(ctx.current_image, "RGB").save(dest, "webp", quality=quality)
            logger.info("Wrote %s", dest.name)
            return True

        if ctx.metadata.get("compose_applied") and ctx.current_image is not None:
            dest = ctx.output_path / f"{stem}.jpg"
            quality = int(ctx.metadata.get("jpeg_quality", 94))
            Image.fromarray(ctx.current_image, "RGB").save(dest, "JPEG", quality=quality)
            logger.info("Wrote composed JPEG %s", dest.name)
            return True

        if ctx.current_rgba is not None:
            dest = ctx.output_path / f"{stem}.png"
            Image.fromarray(ctx.current_rgba, "RGBA").save(dest)
            logger.info("Wrote RGBA %s", dest.name)
            return True

        if ctx.current_image is None:
            logger.warning("Skip %s: no raster to write.", ctx.input_path.name)
            return False

        if write_format == "webp":
            dest = ctx.output_path / f"{stem}.webp"
            quality = int(ctx.metadata.get("webp_quality", 100))
            Image.fromarray(ctx.current_image, "RGB").save(dest, "webp", quality=quality)
        else:
            dest = ctx.output_path / f"{stem}.jpg"
            quality = int(ctx.metadata.get("jpeg_quality", 94))
            Image.fromarray(ctx.current_image, "RGB").save(dest, "JPEG", quality=quality)

        logger.info("Wrote %s", dest.name)
        return True
    except OSError as exc:
        logger.warning("Skip %s: output failed (%s).", ctx.input_path.name, exc)
        return False


def run_pipeline(*, steps: list[str] | None = None) -> PipelineRunSummary:
    """
    Execute the pipeline for every eligible file under ``IMAGE_UTIL_INPUT_DIR``.

    Environment must already be loaded (via :func:`load_image_utility_env`).
    Logging should already be configured by the caller when using a dedicated log file.
    """
    logger = logging.getLogger(__name__)
    step_list = list(steps) if steps is not None else _parse_steps_from_env()
    if not step_list:
        logger.error("%s is empty; provide at least one phase.", ENV_PIPELINE_STEPS)
        return PipelineRunSummary(exit_code=1, processed=0, skipped=0)

    try:
        phases = resolve_pipeline_phases(step_list)
    except KeyError as exc:
        logger.error("%s", exc)
        return PipelineRunSummary(exit_code=1, processed=0, skipped=0)

    input_dir = resolve_dir_from_env(ENV_INPUT_DIR)
    output_base = resolve_dir_from_env(ENV_OUTPUT_DIR)
    max_files = parse_positive_int_env(ENV_MAX_FILES)

    if input_dir is None:
        logger.error("%s is not set in .env.", ENV_INPUT_DIR)
        return PipelineRunSummary(exit_code=1, processed=0, skipped=0)
    if output_base is None:
        logger.error("%s is not set in .env.", ENV_OUTPUT_DIR)
        return PipelineRunSummary(exit_code=1, processed=0, skipped=0)

    if not input_dir.is_dir():
        logger.error("Not a directory: %s", input_dir)
        return PipelineRunSummary(exit_code=1, processed=0, skipped=0)

    _ensure_debug_layout()

    files = sorted_image_files(input_dir)
    if _thumbnail_mode_active(step_list):
        files = _filter_thumbnail_files(files)

    out_dir = _output_directory_for_job(output_base, step_list)

    ok_count = 0
    skipped_count = 0

    logger.info("Pipeline steps: %s", ", ".join(p.phase_name for p in phases))
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", out_dir)

    for source_path in files:
        if max_files is not None and (ok_count + skipped_count) >= max_files:
            logger.info("Reached %s=%s, stopping.", ENV_MAX_FILES, max_files)
            break

        logger.info("File: %s", source_path.name)
        ctx = PipelineContext(
            input_path=source_path,
            output_path=out_dir,
            metadata={"write_format": "jpeg"},
            debug={},
        )

        if not _load_source_into_context(ctx, logger):
            skipped_count += 1
            continue

        failed = False
        for phase in phases:
            logger.info("Phase: %s", phase.phase_name)
            try:
                phase.process(ctx)
            except OSError as exc:
                logger.warning("Phase %s failed for %s: %s", phase.phase_name, source_path.name, exc)
                failed = True
                break
            except Exception as exc:
                logger.warning(
                    "Phase %s error for %s: %s", phase.phase_name, source_path.name, exc, exc_info=True
                )
                failed = True
                break

        if failed:
            skipped_count += 1
            continue

        if ctx.metadata.get("compress_exported"):
            ok_count += 1
        elif _write_final_output(ctx, logger):
            ok_count += 1
        else:
            skipped_count += 1

    logger.info("Done. %s image(s) processed, %s skipped.", ok_count, skipped_count)
    return PipelineRunSummary(exit_code=0, processed=ok_count, skipped=skipped_count)


def run() -> int:
    """Default CLI job: load env, log to ``pipeline.log``, run env-configured steps."""
    load_image_utility_env()
    init_job_logging("pipeline.log")
    return run_pipeline().exit_code
