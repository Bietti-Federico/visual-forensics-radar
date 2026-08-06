"""RandomForest multiclass entity classifier.

`DictVectorizer.transform` silently ignores categorical values it never saw
during `fit` (e.g. a producer string from an entity outside the training
set) rather than raising — exactly the graceful degradation wanted here: an
unfamiliar document should read as low-confidence across every known class,
not crash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

from pdf_forensics.application.entity_identification.ports import FeatureValue
from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction


class RandomForestEntityClassifier:
    classifier_id = "random_forest_entity"

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state
        self._vectorizer: DictVectorizer | None = None
        self._estimator: Any = None

    def fit(
        self, feature_vectors: Sequence[Mapping[str, FeatureValue]], entity_labels: Sequence[str]
    ) -> None:
        self._vectorizer = DictVectorizer(sparse=False)
        matrix = self._vectorizer.fit_transform([dict(vector) for vector in feature_vectors])
        self._estimator = RandomForestClassifier(random_state=self._random_state)
        self._estimator.fit(matrix, list(entity_labels))

    def predict(self, feature_vector: Mapping[str, FeatureValue]) -> EntityPrediction:
        if self._vectorizer is None or self._estimator is None:
            raise RuntimeError(f"{self.classifier_id} has not been fit yet.")

        matrix = self._vectorizer.transform([dict(feature_vector)])
        probabilities = self._estimator.predict_proba(matrix)[0]
        probability_by_entity = {
            str(entity_class): float(probability)
            for entity_class, probability in zip(
                self._estimator.classes_, probabilities, strict=True
            )
        }
        predicted_entity = str(self._estimator.predict(matrix)[0])
        return EntityPrediction(
            classifier_id=self.classifier_id,
            predicted_entity=predicted_entity,
            confidence=probability_by_entity[predicted_entity],
            probabilities=probability_by_entity,
        )
