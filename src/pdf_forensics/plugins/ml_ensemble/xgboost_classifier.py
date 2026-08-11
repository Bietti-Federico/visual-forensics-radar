"""XGBoost classifier with TreeExplainer SHAP support.

Of the spec's three gradient-boosting libraries (CatBoost, LightGBM,
XGBoost), only this one is implemented — confirmed with the user.
CatBoost/LightGBM are functionally overlapping with this and each a heavy
native-build dependency; explicitly deferred, not faked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import shap
from xgboost import XGBClassifier

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin


class XGBoostModel(SklearnClassifierPlugin):
    model_id = "xgboost"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._random_state = random_state

    def _build_estimator(self, labels: Sequence[bool]) -> Any:
        return XGBClassifier(random_state=self._random_state, eval_metric="logloss")

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return shap.TreeExplainer(estimator)
