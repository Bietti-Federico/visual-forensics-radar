"""Trains every configured entity classifier on a batch of FeatureSets labeled
by which known entity produced each document.

Optional `sample_weight` (same length as `feature_sets`) — see
`application/ml_shared/sample_weighting.py` — lets some samples count more
than others, same as Modules 5/6.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.entity_identification.feature_vectorizer import (
    select_entity_features,
)
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.ml_shared.sample_weighting import apply_sample_weight
from pdf_forensics.domain.features.feature_set import FeatureSet


class TrainEntityClassifierUseCase:
    def __init__(self, classifiers: Sequence[EntityClassifierPlugin]) -> None:
        self._classifiers = tuple(classifiers)

    def execute(
        self,
        feature_sets: Sequence[FeatureSet],
        entity_labels: Sequence[str],
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        vectors = [select_entity_features(feature_set) for feature_set in feature_sets]
        pairs = apply_sample_weight(list(zip(vectors, entity_labels, strict=True)), sample_weight)
        weighted_vectors = [vector for vector, _ in pairs]
        weighted_labels = [label for _, label in pairs]
        for classifier in self._classifiers:
            classifier.fit(weighted_vectors, weighted_labels)
