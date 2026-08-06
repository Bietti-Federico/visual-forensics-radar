"""Shared fit/score plumbing for the three sklearn-based detectors.

`IsolationForest`, `OneClassSVM`, and `LocalOutlierFactor(novelty=True)` all
share the same `decision_function` sign convention (positive = inlier,
negative = outlier) — the sign-flip to "higher = more anomalous" and the
`< 0` anomaly threshold live here once, so each concrete detector only
supplies which estimator to build.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.feature_extraction import DictVectorizer

from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore


class SklearnAnomalyDetector:
    detector_id: str

    def __init__(self) -> None:
        self._vectorizer: DictVectorizer | None = None
        self._estimator: Any = None

    def fit(self, feature_vectors: Sequence[Mapping[str, float]]) -> None:
        self._vectorizer = DictVectorizer(sparse=False)
        matrix = self._vectorizer.fit_transform(list(feature_vectors))
        self._estimator = self._build_estimator(len(feature_vectors))
        self._estimator.fit(matrix)

    def score(self, feature_vector: Mapping[str, float]) -> AnomalyScore:
        if self._vectorizer is None or self._estimator is None:
            raise RuntimeError(f"{self.detector_id} has not been fit yet.")
        matrix = self._vectorizer.transform([dict(feature_vector)])
        decision = float(self._estimator.decision_function(matrix)[0])
        return AnomalyScore(detector_id=self.detector_id, score=-decision, is_anomaly=decision < 0)

    def _build_estimator(self, n_samples: int) -> Any:
        raise NotImplementedError
