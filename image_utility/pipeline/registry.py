"""Registered pipeline phases (lookup by name, no hard-coded dispatch chains)."""

from __future__ import annotations

from .contracts import PipelinePhase

PHASE_REGISTRY: dict[str, PipelinePhase] = {}

_default_phases_initialized = False


def register_phase(phase: PipelinePhase) -> None:
    key = phase.phase_name.strip().lower()
    PHASE_REGISTRY[key] = phase


def _ensure_default_phases() -> None:
    """
    Load built-in phases on first use so importing ``contracts``/``runner`` does not cycle.

    Skips registration when ``PHASE_REGISTRY`` is already non-empty (e.g. tests or custom plugins).
    """
    global _default_phases_initialized
    if _default_phases_initialized:
        return
    _default_phases_initialized = True
    if PHASE_REGISTRY:
        return

    from image_utility.compress.phase import CompressPhase
    from image_utility.isolate.phase import IsolatePhase

    register_phase(CompressPhase())
    register_phase(IsolatePhase())


def resolve_pipeline_phases(step_names: list[str]) -> list[PipelinePhase]:
    _ensure_default_phases()
    out: list[PipelinePhase] = []
    for raw in step_names:
        name = raw.strip().lower()
        if not name:
            continue
        phase = PHASE_REGISTRY.get(name)
        if phase is None:
            valid = ", ".join(sorted(PHASE_REGISTRY))
            raise KeyError(f"Unknown pipeline phase '{raw}'. Registered phases: {valid}")
        out.append(phase)
    return out
