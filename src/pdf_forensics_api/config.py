"""Filesystem locations for the API's persistent state.

The training corpus and the current model bundle are the only two things
allowed to grow on disk (see `docs/DOCUMENTACION.md`) — `/verify` never writes
anything. Read fresh on every call (not cached at import time) so tests can
point them at a `tmp_path` by setting the environment variables before the
app's `lifespan` startup runs.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRAINING_CORPUS_DIR_ENV = "PDF_FORENSICS_TRAINING_CORPUS_DIR"
_MODEL_STORE_PATH_ENV = "PDF_FORENSICS_MODEL_STORE_PATH"
_LOG_PATH_ENV = "PDF_FORENSICS_LOG_PATH"


def get_training_corpus_dir() -> Path:
    return Path(os.environ.get(_TRAINING_CORPUS_DIR_ENV, "training_corpus"))


def get_model_store_path() -> Path:
    return Path(os.environ.get(_MODEL_STORE_PATH_ENV, "model_store.joblib"))


def get_log_path() -> Path:
    return Path(os.environ.get(_LOG_PATH_ENV, "pdf_forensics_api.log"))
