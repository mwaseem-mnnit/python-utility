"""Isolate phase package — lazy exports avoid pulling rembg until isolate runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .components import ComponentInfo, select_product_label
from .config import IsolateConfig, load_isolate_config

__all__ = [
    "ComponentInfo",
    "IsolateConfig",
    "IsolatePhase",
    "load_isolate_config",
    "process_isolate",
    "select_product_label",
]

if TYPE_CHECKING:
    from .phase import IsolatePhase
    from .processor import process_isolate


def __getattr__(name: str) -> Any:
    if name == "IsolatePhase":
        from .phase import IsolatePhase as _IsolatePhase

        return _IsolatePhase
    if name == "process_isolate":
        from .processor import process_isolate as _process_isolate

        return _process_isolate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
