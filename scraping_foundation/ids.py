from __future__ import annotations

"""Sequential product IDs: ``p_<6-digit>`` starting at ``p_100001`` (in-memory state)."""

_START = 100_001
_counter: int | None = None


def reset_product_id_counter(start: int = _START) -> None:
    """Reset the in-memory counter (e.g. for tests). ``start`` must be >= 100001."""
    global _counter
    if start < _START:
        raise ValueError(f"start must be >= {_START}, got {start}")
    _counter = start


def peek_next_product_id() -> str:
    """Return the next ID that will be issued, without consuming it."""
    n = _START if _counter is None else _counter
    return _format(n)


def next_product_id() -> str:
    """Return the next ID and advance the counter."""
    global _counter
    if _counter is None:
        _counter = _START
    pid = _format(_counter)
    _counter += 1
    return pid


def _format(n: int) -> str:
    return f"p_{n:06d}"
