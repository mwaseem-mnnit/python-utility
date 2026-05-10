"""Isolate phase — semantic product isolation (RGBA + alpha mask)."""

from __future__ import annotations

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .processor import process_isolate


class IsolatePhase(PipelinePhase):
    """rembg + CC/heuristics → transparent RGBA; no resize or composition."""

    phase_name = "isolate"

    def process(self, context: PipelineContext) -> PipelineContext:
        return process_isolate(context)
