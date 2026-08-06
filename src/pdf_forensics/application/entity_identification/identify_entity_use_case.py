"""Identifies which known entity likely produced one document, per configured classifier."""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.entity_identification.feature_vectorizer import (
    select_entity_features,
)
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet


class IdentifyEntityUseCase:
    def __init__(self, classifiers: Sequence[EntityClassifierPlugin]) -> None:
        self._classifiers = tuple(classifiers)

    def execute(self, feature_set: FeatureSet) -> EntityIdentificationReport:
        vector = select_entity_features(feature_set)
        predictions = [classifier.predict(vector) for classifier in self._classifiers]
        return EntityIdentificationReport(predictions=predictions)
