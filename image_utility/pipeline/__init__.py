"""
Image pipeline package.

Import heavy entrypoints from submodules (e.g. ``image_utility.pipeline.runner``) so that
``from image_utility.pipeline.contracts import …`` never pulls in the orchestrator or registry.

For convenience, ``run``, ``run_pipeline``, and ``PipelineRunSummary`` are also exposed via lazy
``__getattr__`` (PEP 562) without eager imports at package load time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "PipelineRunSummary",
    "run",
    "run_pipeline",
]

if TYPE_CHECKING:
    from .runner import PipelineRunSummary as PipelineRunSummary
    from .runner import run as run
    from .runner import run_pipeline as run_pipeline


def __getattr__(name: str) -> Any:
    if name == "PipelineRunSummary":
        from .runner import PipelineRunSummary as prs

        return prs
    if name == "run":
        from .runner import run as r

        return r
    if name == "run_pipeline":
        from .runner import run_pipeline as rp

        return rp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
