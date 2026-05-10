"""Phase execution contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .context import PipelineContext


class PipelinePhase(ABC):
    """One pipeline step; operates on and returns the shared :class:`PipelineContext`."""

    phase_name: str

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Transform ``context`` in place and return it."""

