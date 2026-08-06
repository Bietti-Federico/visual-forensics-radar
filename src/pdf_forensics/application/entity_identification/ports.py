"""The port entity-identification use cases depend on.

`plugins/entity_identification/` implements it. Same stateful shape as
Module 6's `SupervisedModelPlugin`: a classifier must
be `fit()` on a labeled batch before `predict()` can be called, and every
implementation raises clearly if called first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction

FeatureValue = float | str


class EntityClassifierPlugin(Protocol):
    classifier_id: str

    def fit(
        self, feature_vectors: Sequence[Mapping[str, FeatureValue]], entity_labels: Sequence[str]
    ) -> None:
        """Fit on a batch labeled with the entity that produced each document."""
        ...

    def predict(self, feature_vector: Mapping[str, FeatureValue]) -> EntityPrediction:
        """Predict for one feature vector. Raises if called before `fit()`."""
        ...
