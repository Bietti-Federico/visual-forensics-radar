"""Trains every configured supervised model on a labeled batch of FeatureSets.

Natural consumer: Training Data Ingestion's `TrainingDataset` —
`labels = [sample.source.is_original for sample in dataset.samples]`,
`feature_sets = [sample.features for sample in dataset.samples]`.

Optional `sample_weight` (same length as `feature_sets`) lets some samples
count more than others — e.g. genuine real documents weighted above
synthetic field-substituted variants generated from them while a training
corpus is still small. See `application/ml_shared/sample_weighting.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.application.ml_shared.sample_weighting import apply_sample_weight
from pdf_forensics.domain.features.feature_set import FeatureSet


class TrainModelsUseCase:
    def __init__(self, models: Sequence[SupervisedModelPlugin]) -> None:
        self._models = tuple(models)

    def execute(
        self,
        feature_sets: Sequence[FeatureSet],
        labels: Sequence[bool],
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        vectors = [select_numeric_features(feature_set) for feature_set in feature_sets]
        pairs = apply_sample_weight(list(zip(vectors, labels, strict=True)), sample_weight)
        weighted_vectors = [vector for vector, _ in pairs]
        weighted_labels = [label for _, label in pairs]
        for model in self._models:
            model.fit(weighted_vectors, weighted_labels)
