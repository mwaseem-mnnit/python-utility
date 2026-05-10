"""Isolate phase — semantic product isolation (RGBA + alpha mask)."""

from __future__ import annotations

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .processor import process_isolate


class IsolatePhase(PipelinePhase):
    """rembg + connected components + cleanup → transparent RGBA; no resize or composition."""

    phase_name = "isolate"

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Run isolation: segment → analyze components → clean → refine → optional debug.

        Delegates to :func:`process_isolate` for file-scope orchestration.
        """
        return process_isolate(context)
