"""Shared mutable state for sequential pipeline phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PipelineContext:
    """
    Phase-oriented execution state. Mutated in place as each phase runs.

    ``output_path`` is the output **directory** for the current job/file batch.
    """

    input_path: Path
    output_path: Path
    current_image: np.ndarray | None = None
    current_rgba: np.ndarray | None = None
    alpha_mask: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
