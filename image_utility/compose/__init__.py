"""Ecommerce compose package (lazy heavy imports)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import ComposeConfig, load_compose_config

__all__ = [
    "ComposeConfig",
    "ComposePhase",
    "load_compose_config",
    "process_compose",
]

if TYPE_CHECKING:
    from .phase import ComposePhase
    from .processor import process_compose


def __getattr__(name: str) -> Any:
    if name == "ComposePhase":
        from .phase import ComposePhase as _ComposePhase

        return _ComposePhase
    if name == "process_compose":
        from .processor import process_compose as _process_compose

        return _process_compose
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
