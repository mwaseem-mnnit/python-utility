"""
Load ``site_2/.env`` and initialise :mod:`app_logging` for site_2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PACKAGE_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app_logging import init_logging

_ENV_FILE = PACKAGE_DIR / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE)

init_logging(default_filename="app.log")
