"""Polish phase — subtle RGB refinement for ecommerce output."""

from __future__ import annotations

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .processor import process_polish


class PolishPhase(PipelinePhase):
    """Mild contrast, brightness, LAB clarity, and unsharp mask on composed RGB."""

    phase_name = "polish"

    def process(self, context: PipelineContext) -> PipelineContext:
        return process_polish(context)
