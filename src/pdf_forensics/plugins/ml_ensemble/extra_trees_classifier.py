"""Extra Trees classifier with TreeExplainer SHAP support."""

from __future__ import annotations

from typing import Any

import shap
from sklearn.ensemble import ExtraTreesClassifier

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin


class ExtraTreesModel(SklearnClassifierPlugin):
    model_id = "extra_trees"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._random_state = random_state

    def _build_estimator(self) -> Any:
        return ExtraTreesClassifier(random_state=self._random_state)

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return shap.TreeExplainer(estimator)
