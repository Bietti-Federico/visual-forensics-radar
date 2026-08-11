"""Stacking ensemble over RandomForest/ExtraTrees/LogisticRegression/XGBoost.

SHAP is explicitly deferred for this model, not faked: a heterogeneous
stacked combiner needs `shap.KernelExplainer` (model-agnostic, but slow and
needs careful background-sampling tuning to be reliable) — `_build_explainer`
returns `None`, and `explain()` (in `_sklearn_classifier_base.py`) raises
`NotImplementedError` for it.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pdf_forensics.plugins.ml_ensemble._sklearn_classifier_base import SklearnClassifierPlugin

# StackingClassifier internally cross-validates each base estimator with
# StratifiedKFold(cv), which requires every class to have at least `cv`
# members. Its sklearn default (cv=5) crashes at this platform's documented
# minimum training size (5 genuine / 3 fraud) since the minority class has
# only 3 members. Capping cv at the smallest class's size keeps it fittable
# down to 2 members per class (StackingClassifier's own hard floor).
_MAX_CV = 5
_MIN_CV = 2


class StackingEnsembleModel(SklearnClassifierPlugin):
    model_id = "stacking_ensemble"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._random_state = random_state

    def _build_estimator(self, labels: Sequence[bool]) -> Any:
        smallest_class_size = min(Counter(labels).values())
        cv = max(_MIN_CV, min(_MAX_CV, smallest_class_size))
        estimators = [
            ("random_forest", RandomForestClassifier(random_state=self._random_state)),
            ("extra_trees", ExtraTreesClassifier(random_state=self._random_state)),
            ("logistic_regression", LogisticRegression(max_iter=1000)),
            ("xgboost", XGBClassifier(random_state=self._random_state, eval_metric="logloss")),
        ]
        return StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=cv
        )

    def _build_explainer(self, estimator: Any, background: Any) -> Any:
        return None
