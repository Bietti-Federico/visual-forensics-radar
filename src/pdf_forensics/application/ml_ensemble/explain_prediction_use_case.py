"""Explains one FeatureSet's prediction via SHAP, for every model that supports it.

Models whose `explain()` raises `NotImplementedError` (the two ensemble
combiners — see `plugins/ml_ensemble/stacking_ensemble.py` and
`voting_ensemble.py`) are skipped, not treated as an error: "no explanation
available for this model" is an expected, documented outcome, not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation


class ExplainPredictionUseCase:
    def __init__(self, models: Sequence[SupervisedModelPlugin]) -> None:
        self._models = tuple(models)

    def execute(self, feature_set: FeatureSet) -> list[ShapExplanation]:
        vector = select_numeric_features(feature_set)
        explanations = []
        for model in self._models:
            try:
                explanations.append(model.explain(vector))
            except NotImplementedError:
                continue
        return explanations
