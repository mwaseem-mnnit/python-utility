"""Polish package (lazy heavy imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import PolishConfig, load_polish_config

__all__ = [
    "PolishConfig",
    "PolishPhase",
    "load_polish_config",
    "process_polish",
]

if TYPE_CHECKING:
    from .phase import PolishPhase
    from .processor import process_polish


def __getattr__(name: str) -> Any:
    if name == "PolishPhase":
        from .phase import PolishPhase as _PolishPhase

        return _PolishPhase
    if name == "process_polish":
        from .processor import process_polish as _process_polish

        return _process_polish
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
