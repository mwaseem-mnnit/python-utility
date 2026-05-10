"""Compress phase — minimal surface for registry import (no runner / job orchestration)."""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

WEBP_SIZE = 950
THUMBNAIL_SIZE = 420


class CompressPhase(PipelinePhase):
    """Resize source RGB to square WebP dimensions; runner performs final encode."""

    phase_name = "compress"

    def process(self, context: PipelineContext) -> PipelineContext:
        if context.current_image is None:
            raise OSError("compress requires current_image on context.")

        is_thumb = os.getenv("IMAGE_UTIL_THUMBNAIL", "").strip() == "1"
        size = THUMBNAIL_SIZE if is_thumb else WEBP_SIZE
        pil = Image.fromarray(context.current_image)
        resized = pil.resize((size, size), Image.Resampling.LANCZOS)
        context.current_image = np.array(resized)
        context.metadata["write_format"] = "webp"
        context.metadata["webp_quality"] = 100
        return context
