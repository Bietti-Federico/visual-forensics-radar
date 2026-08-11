"""Thin joblib-based persistence for fitted plugin instances.

A fitted `AnomalyDetectorPlugin`/`SupervisedModelPlugin` instance is an
ordinary picklable Python object under the hood — a `DictVectorizer`, a
scikit-learn/XGBoost estimator, a SHAP explainer, or (the autoencoder) a small
`torch.nn.Module`. joblib.dump/load handles all of them directly with no
per-plugin serialization code; this module is deliberately just a thin,
versioned wrapper around that, not a per-plugin (de)serializer.

TODO(deploy): storage footprint is not optimized yet — `joblib.dump` is
called here with no `compress` level, and the bundle keeps every model's
full estimator (e.g. RandomForest/ExtraTrees keep every tree). Revisit once
there's a real test suite and a concrete deployment target with tight disk
constraints: candidates are `joblib.dump(..., compress=3)` (fast, no new
dependency), pruning/quantizing tree ensembles, or dropping SHAP explainer
background data from the persisted bundle and rebuilding it lazily on load.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib

#: 2.0.0: switched from one global detector/model set to a per-entity bundle
#: (`application/model_persistence/entity_model_bundle.py`).
#: 3.0.0: `EntityModelBundle` gained `invariants` (auto-mined per-entity
#: template invariants, replacing the hand-written `entity_template_mismatch`
#: rule). Both bumps are incompatible on purpose, not silently misread.
MODEL_STORE_SCHEMA_VERSION = "3.0.0"


def save_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and atomically rename it into place, so a
    # crash or a concurrent /retrain mid-write can never leave a truncated or
    # half-written model store behind for a subsequent load to trip over.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        joblib.dump({"schema_version": MODEL_STORE_SCHEMA_VERSION, **bundle}, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No model store found at {path}")

    bundle: dict[str, Any] = joblib.load(path)
    schema_version = bundle.get("schema_version")
    if schema_version != MODEL_STORE_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {schema_version!r}, "
            f"expected {MODEL_STORE_SCHEMA_VERSION!r}"
        )
    return bundle
