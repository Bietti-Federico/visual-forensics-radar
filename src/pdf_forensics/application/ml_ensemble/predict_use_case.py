"""Predicts one FeatureSet against every already-fitted supervised model."""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport


class PredictUseCase:
    def __init__(self, models: Sequence[SupervisedModelPlugin]) -> None:
        self._models = tuple(models)

    def execute(self, feature_set: FeatureSet) -> MlEnsembleReport:
        vector = select_numeric_features(feature_set)
        predictions = [model.predict(vector) for model in self._models]
        return MlEnsembleReport(predictions=predictions)
