"""Stacking ensemble over RandomForest/ExtraTrees/LogisticRegression/XGBoost.

SHAP is explicitly deferred for this model, not faked: a heterogeneous
stacked combiner needs `shap.KernelExplainer` (model-agnostic, but slow and
needs careful background-sampling tuning to be reliable) — `_build_explainer`
returns `None`, and `explain()` (in `_sklearn_classifier_base.py`) raises
`NotImplementedError` for it.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin


class StackingEnsembleModel(SklearnClassifierPlugin):
    model_id = "stacking_ensemble"

    def _build_estimator(self) -> Any:
        estimators = [
            ("random_forest", RandomForestClassifier(random_state=42)),
            ("extra_trees", ExtraTreesClassifier(random_state=42)),
            ("logistic_regression", LogisticRegression(max_iter=1000)),
            ("xgboost", XGBClassifier(random_state=42, eval_metric="logloss")),
        ]
        return StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(max_iter=1000)
        )

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return None
