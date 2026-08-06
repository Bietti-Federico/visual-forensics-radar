"""Isolation Forest anomaly detector (sklearn.ensemble.IsolationForest)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import IsolationForest

from pdf_forensics.plugins.anomaly_detection._sklearn_base import SklearnAnomalyDetector


class IsolationForestDetector(SklearnAnomalyDetector):
    detector_id = "isolation_forest"

    def __init__(self, random_state: int = 42) -> None:
        super().__init__()
        self._random_state = random_state

    def _build_estimator(self, n_samples: int) -> Any:
        return IsolationForest(random_state=self._random_state)
