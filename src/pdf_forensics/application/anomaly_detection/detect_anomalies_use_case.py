"""Scores one FeatureSet against every already-fitted anomaly detector."""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.features.feature_set import FeatureSet


class DetectAnomaliesUseCase:
    def __init__(self, detectors: Sequence[AnomalyDetectorPlugin]) -> None:
        self._detectors = tuple(detectors)

    def execute(self, feature_set: FeatureSet) -> AnomalyDetectionReport:
        vector = select_numeric_features(feature_set)
        scores = [detector.score(vector) for detector in self._detectors]
        return AnomalyDetectionReport(scores=scores)
