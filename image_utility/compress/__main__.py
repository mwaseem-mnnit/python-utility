"""
Run as: ``python -m image_utility.compress`` (recommended), or execute this file
directly in an IDE (repo root is added to ``sys.path``).
"""

from __future__ import annotations

import sys
from pathlib import Path


def _main() -> int:
    if __package__ in (None, ""):
        # Executed as a script (e.g. PyCharm "Run"): no package context for relative imports.
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from image_utility.compress.pipeline import main as run

        return run()
    from .pipeline import main as run

    return run()


if __name__ == "__main__":
    raise SystemExit(_main())
