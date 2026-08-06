"""The complete set of entity predictions produced by one identification run.

Same shape as Module 6's `MlEnsembleReport` — one prediction per configured
classifier, `by_classifier` for lookup, `len`/iteration for the common case
of a single default classifier.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction


@dataclass(slots=True)
class EntityIdentificationReport:
    predictions: list[EntityPrediction] = field(default_factory=list)

    def by_classifier(self, classifier_id: str) -> EntityPrediction | None:
        for prediction in self.predictions:
            if prediction.classifier_id == classifier_id:
                return prediction
        return None

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self) -> Iterator[EntityPrediction]:
        return iter(self.predictions)
