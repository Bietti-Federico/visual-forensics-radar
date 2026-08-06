"""Logistic Regression classifier with LinearExplainer SHAP support.

`LinearExplainer` needs background data to compute expected feature values
against — the training batch itself (`background`, passed in by
`_sklearn_classifier_base.py`'s `fit()`) serves as that masker.
"""

from __future__ import annotations

from typing import Any

import shap
from sklearn.linear_model import LogisticRegression

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin


class LogisticRegressionModel(SklearnClassifierPlugin):
    model_id = "logistic_regression"

    def _build_estimator(self) -> Any:
        return LogisticRegression(max_iter=1000)

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return shap.LinearExplainer(estimator, background)
