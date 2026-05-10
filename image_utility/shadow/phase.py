"""Shadow phase — subtle ecommerce grounding."""

from __future__ import annotations

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .processor import process_shadow


class ShadowPhase(PipelinePhase):
    """Soft contact shadow under the composed product (conservative, non-directional)."""

    phase_name = "shadow"

    def process(self, context: PipelineContext) -> PipelineContext:
        return process_shadow(context)
