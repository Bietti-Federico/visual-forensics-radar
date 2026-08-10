"""Basic file logging for the API process.

One rotating log file, configured once at process startup — enough to
track down a production error after the fact without needing a log
aggregation stack. `get_log_path()` follows the same "read fresh, env-var
overridable" pattern as the training corpus/model store paths, so tests can
point it at a `tmp_path` too.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from pdf_forensics_api.config import get_log_path

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (e.g. re-entering lifespan in tests).

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        get_log_path(), maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
