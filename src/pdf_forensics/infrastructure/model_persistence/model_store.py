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

from pathlib import Path
from typing import Any

import joblib

MODEL_STORE_SCHEMA_VERSION = "1.0.0"


def save_bundle(path: str | Path, bundle: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": MODEL_STORE_SCHEMA_VERSION, **bundle}, path)


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
