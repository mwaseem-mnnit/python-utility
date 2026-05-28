"""Isolate semantic substage sequencing.

Orchestration is implemented in :func:`image_utility.isolate.processor.process_isolate`.

Stage order: decomposition → filtering → ranking → grouping → suppression → refinement.
"""

from __future__ import annotations

from image_utility.isolate.processor import process_isolate

__all__ = ["process_isolate"]
