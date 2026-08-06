"""Fits every configured anomaly detector on a reference batch of FeatureSets.

Deliberately independent of any fraud label — a `Sequence[FeatureSet]` is all
this needs, matching the platform's brief ("anomaly detection independent
from fraud labels"). A natural source for that batch is the Training Data
Ingestion module's `TrainingDataset.samples[i].features`.

Optional `sample_weight` (same length as `feature_sets`) lets some samples
count more than others — e.g. genuine real documents weighted above
synthetic field-substituted variants generated from them while a training
corpus is still small. See `application/ml_shared/sample_weighting.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.application.ml_shared.sample_weighting import apply_sample_weight
from pdf_forensics.domain.features.feature_set import FeatureSet


class FitAnomalyDetectorsUseCase:
    def __init__(self, detectors: Sequence[AnomalyDetectorPlugin]) -> None:
        self._detectors = tuple(detectors)

    def execute(
        self,
        feature_sets: Sequence[FeatureSet],
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        vectors = [select_numeric_features(feature_set) for feature_set in feature_sets]
        vectors = apply_sample_weight(vectors, sample_weight)
        for detector in self._detectors:
            detector.fit(vectors)
