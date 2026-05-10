"""Shadow package (lazy heavy imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ShadowConfig, load_shadow_config

__all__ = [
    "ShadowConfig",
    "ShadowPhase",
    "load_shadow_config",
    "process_shadow",
]

if TYPE_CHECKING:
    from .phase import ShadowPhase
    from .processor import process_shadow


def __getattr__(name: str) -> Any:
    if name == "ShadowPhase":
        from .phase import ShadowPhase as _ShadowPhase

        return _ShadowPhase
    if name == "process_shadow":
        from .processor import process_shadow as _process_shadow

        return _process_shadow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
