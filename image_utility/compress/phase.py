"""Compress/Export phase — multi-format delivery assets (WebP, JPEG, thumbnail).

This is the ONLY phase that performs compression/format conversion.
No visual modifications allowed here.
"""

from __future__ import annotations

import logging

from PIL import Image

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .config import load_compress_config

LOGGER = logging.getLogger(__name__)


class CompressPhase(PipelinePhase):
    """Generate delivery-ready ecommerce assets: WebP listing, JPEG full-res, thumbnail."""

    phase_name = "compress"

    def process(self, context: PipelineContext) -> PipelineContext:
        cfg = load_compress_config()

        if context.current_image is None:
            raise OSError("compress requires current_image on context.")

        stem = context.input_path.stem
        out_dir = context.output_path
        out_dir.mkdir(parents=True, exist_ok=True)

        pil = Image.fromarray(context.current_image)
        src_w, src_h = pil.size
        written: list[str] = []

        if cfg.is_thumbnail_only:
            # Thumbnail-only batch mode
            thumb = pil.resize(
                (cfg.thumbnail_size, cfg.thumbnail_size), Image.Resampling.LANCZOS
            )
            dest = out_dir / f"{stem}.webp"
            thumb.save(str(dest), "webp", quality=cfg.thumbnail_quality)
            written.append(dest.name)
            LOGGER.info("[compress] wrote thumbnail %s (%dpx)", dest.name, cfg.thumbnail_size)
        else:
            # Full export: emit multiple formats
            if cfg.emit_jpeg:
                if cfg.jpeg_size > 0 and (src_w != cfg.jpeg_size or src_h != cfg.jpeg_size):
                    jpeg_img = pil.resize(
                        (cfg.jpeg_size, cfg.jpeg_size), Image.Resampling.LANCZOS
                    )
                else:
                    jpeg_img = pil
                dest = out_dir / f"{stem}.jpg"
                jpeg_img.save(str(dest), "JPEG", quality=cfg.jpeg_quality)
                written.append(dest.name)
                LOGGER.info("[compress] wrote JPEG %s (q=%d)", dest.name, cfg.jpeg_quality)

            if cfg.emit_webp:
                webp_img = pil.resize(
                    (cfg.webp_size, cfg.webp_size), Image.Resampling.LANCZOS
                )
                dest = out_dir / f"{stem}.webp"
                webp_img.save(str(dest), "webp", quality=cfg.webp_quality)
                written.append(dest.name)
                LOGGER.info("[compress] wrote WebP %s (%dpx q=%d)", dest.name, cfg.webp_size, cfg.webp_quality)

            if cfg.emit_thumbnail:
                thumb = pil.resize(
                    (cfg.thumbnail_size, cfg.thumbnail_size), Image.Resampling.LANCZOS
                )
                thumb_dir = out_dir / "thumbnail"
                thumb_dir.mkdir(parents=True, exist_ok=True)
                dest = thumb_dir / f"{stem}.webp"
                thumb.save(str(dest), "webp", quality=cfg.thumbnail_quality)
                written.append(f"thumbnail/{dest.name}")
                LOGGER.info("[compress] wrote thumbnail %s (%dpx)", dest.name, cfg.thumbnail_size)

        # Mark output as handled by compress phase (skip runner's _write_final_output)
        context.metadata["compress_exported"] = True
        context.metadata["compress_files"] = written
        context.metadata["write_format"] = "__compress_handled__"
        context.debug["compress_formats"] = written

        LOGGER.info("[compress] exported %d asset(s) for %s", len(written), stem)
        return context
