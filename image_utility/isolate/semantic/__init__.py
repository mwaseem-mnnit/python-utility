"""isolate.semantic — optional SAM-based foreground refinement (v3)."""

from __future__ import annotations

from .refinement import apply_semantic_refinement, should_activate_semantic_refinement
from .sam_runner import generate_raw_masks, is_sam_available

__all__ = [
    "apply_semantic_refinement",
    "generate_raw_masks",
    "is_sam_available",
    "should_activate_semantic_refinement",
]
