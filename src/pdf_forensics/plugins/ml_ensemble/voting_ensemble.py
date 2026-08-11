"""Soft-voting ensemble over RandomForest/ExtraTrees/LogisticRegression/XGBoost.

SHAP is explicitly deferred for this model — same reasoning as
`stacking_ensemble.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin


class VotingEnsembleModel(SklearnClassifierPlugin):
    model_id = "voting_ensemble"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._random_state = random_state

    def _build_estimator(self, labels: Sequence[bool]) -> Any:
        estimators = [
            ("random_forest", RandomForestClassifier(random_state=self._random_state)),
            ("extra_trees", ExtraTreesClassifier(random_state=self._random_state)),
            ("logistic_regression", LogisticRegression(max_iter=1000)),
            ("xgboost", XGBClassifier(random_state=self._random_state, eval_metric="logloss")),
        ]
        return VotingClassifier(estimators=estimators, voting="soft")

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return None
