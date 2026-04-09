"""
Shared file logging for python-utility projects.

Call :func:`init_logging` once at process startup (after loading the module's ``.env``).
Use ``LOG_FILE`` in that ``.env`` for the log path; if unset, ``default_filename`` (``app.log``) is used.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = ["init_logging"]


def init_logging(
    *,
    log_file: str | Path | None = None,
    env_var: str = "LOG_FILE",
    default_filename: str = "app.log",
    level: int = logging.INFO,
    format_string: str | None = None,
    also_stdout: bool = False,
) -> Path:
    """
    Configure the root logger to append UTF-8 lines to a single file.

    * ``log_file`` — explicit path; if omitted, ``os.environ[env_var]``; if still empty,
      ``default_filename`` (relative paths are resolved under :func:`os.getcwd`).
    * ``also_stdout`` — mirror logs to stdout (e.g. image compress CLI).

    Returns the resolved log file path. Safe to call again (handlers are replaced).
    """
    raw: str | Path | None = log_file
    if raw is None:
        raw = os.environ.get(env_var, "").strip()
    if not raw:
        raw = default_filename

    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = format_string or "%(asctime)s %(levelname)s %(name)s - %(message)s"
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    root.info("Logging initialized; file=%s", path)
    return path
