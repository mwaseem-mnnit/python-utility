"""Compose phase — ecommerce white-canvas framing."""

from __future__ import annotations

from image_utility.pipeline.contracts import PipelinePhase
from image_utility.pipeline.context import PipelineContext

from .processor import process_compose


class ComposePhase(PipelinePhase):
    """Scale isolated RGBA onto a configurable white canvas with balanced placement."""

    phase_name = "compose"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Isolate RGBA → crop → scale → place → alpha-blend on background."""
        return process_compose(context)
