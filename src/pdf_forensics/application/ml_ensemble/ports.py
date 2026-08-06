"""The port ml_ensemble use cases depend on; `plugins/ml_ensemble/` implements it.

Same stateful-plugin shape as Module 5's `AnomalyDetectorPlugin`: a model
must be `fit()` before `predict()`/`explain()`, and every implementation
raises clearly if called first. `explain()` is allowed to raise
`NotImplementedError` for models that genuinely can't produce a SHAP
explanation this iteration (the two ensemble combiners) — that's a
documented, permanent limitation of those two models, not a "not fit yet" error.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation


class SupervisedModelPlugin(Protocol):
    model_id: str

    def fit(self, feature_vectors: Sequence[Mapping[str, float]], labels: Sequence[bool]) -> None:
        """Fit on a labeled batch. `labels[i]` is `is_original` for `feature_vectors[i]`."""
        ...

    def predict(self, feature_vector: Mapping[str, float]) -> ModelPrediction:
        """Predict for one feature vector. Raises if called before `fit()`."""
        ...

    def explain(self, feature_vector: Mapping[str, float]) -> ShapExplanation:
        """Explain one prediction via SHAP.

        Raises `NotImplementedError` for models that don't support SHAP this
        iteration (see the model's own docstring), or a `RuntimeError` if
        called before `fit()`.
        """
        ...
