"""Anomaly detector plugins.

Explicitly wired (not auto-registered): `default_detectors()` below is the
one place to touch when adding, removing, or swapping a detector.

Persisting fitted detectors to disk lives in
`application/model_persistence/`. Combining multiple detectors' scores into
one signal is `application/risk_report/`.
"""

from __future__ import annotations

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.plugins.anomaly_detection.autoencoder_detector import (
    AutoencoderAnomalyDetector,
)
from pdf_forensics.plugins.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from pdf_forensics.plugins.anomaly_detection.local_outlier_factor_detector import (
    LocalOutlierFactorDetector,
)
from pdf_forensics.plugins.anomaly_detection.one_class_svm_detector import OneClassSVMDetector


def default_detectors() -> tuple[AnomalyDetectorPlugin, ...]:
    return (
        IsolationForestDetector(),
        OneClassSVMDetector(),
        LocalOutlierFactorDetector(),
        AutoencoderAnomalyDetector(),
    )


__all__ = ["default_detectors"]
