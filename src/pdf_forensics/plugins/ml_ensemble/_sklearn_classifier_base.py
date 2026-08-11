"""Shared fit/predict/explain plumbing for sklearn-style binary classifiers.

The positive class (`1`/`True`) is always "manipulated" (`is_original is
False`) — `ModelPrediction.probability` is P(manipulated), consistent with
Module 5's "higher = more anomalous" convention.

`_build_explainer()` may return `None` (used by the two ensemble combiners,
which don't support SHAP this iteration) — `explain()` then raises
`NotImplementedError` rather than attempting to use a missing explainer,
distinct from the `RuntimeError` raised when the model hasn't been fit yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation


class SklearnClassifierPlugin:
    model_id: str

    def __init__(self) -> None:
        self._vectorizer: DictVectorizer | None = None
        self._estimator: Any = None
        self._explainer: Any = None

    def fit(self, feature_vectors: Sequence[Mapping[str, float]], labels: Sequence[bool]) -> None:
        target = [not label for label in labels]  # positive class = "manipulated"
        if len(set(target)) < 2:
            raise ValueError(
                f"{self.model_id} requires both genuine and manipulated examples to fit; "
                f"got only one class."
            )

        self._vectorizer = DictVectorizer(sparse=False)
        matrix = self._vectorizer.fit_transform(list(feature_vectors))

        self._estimator = self._build_estimator(target)
        self._estimator.fit(matrix, target)
        self._explainer = self._build_explainer(self._estimator, matrix)

    def predict(self, feature_vector: Mapping[str, float]) -> ModelPrediction:
        matrix = self._transform(feature_vector)
        probability = float(self._estimator.predict_proba(matrix)[0][1])
        predicted_label = bool(self._estimator.predict(matrix)[0])
        return ModelPrediction(
            model_id=self.model_id, probability=probability, predicted_label=predicted_label
        )

    def explain(self, feature_vector: Mapping[str, float]) -> ShapExplanation:
        matrix = self._transform(feature_vector)
        if self._explainer is None:
            raise NotImplementedError(
                f"{self.model_id} does not support SHAP explanations this iteration."
            )

        explanation = self._explainer(matrix)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[..., -1]  # multi-output explainer: keep the "manipulated" class
        row = values[0]

        base = np.asarray(explanation.base_values)
        base_value = float(base[0][-1]) if base.ndim == 2 else float(base[0])

        assert self._vectorizer is not None
        return ShapExplanation(
            model_id=self.model_id,
            feature_names=tuple(self._vectorizer.get_feature_names_out()),
            shap_values=tuple(float(v) for v in row),
            base_value=base_value,
        )

    def _transform(self, feature_vector: Mapping[str, float]) -> Any:
        if self._vectorizer is None or self._estimator is None:
            raise RuntimeError(f"{self.model_id} has not been fit yet.")
        return self._vectorizer.transform([dict(feature_vector)])

    def _build_estimator(self, labels: Sequence[bool]) -> Any:
        raise NotImplementedError

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        raise NotImplementedError
