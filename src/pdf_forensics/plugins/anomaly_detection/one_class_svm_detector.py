"""One-Class SVM anomaly detector (sklearn.svm.OneClassSVM)."""

from __future__ import annotations

from typing import Any

from sklearn.svm import OneClassSVM

from pdf_forensics.plugins.anomaly_detection._sklearn_base import SklearnAnomalyDetector


class OneClassSVMDetector(SklearnAnomalyDetector):
    detector_id = "one_class_svm"

    def _build_estimator(self, n_samples: int) -> Any:
        return OneClassSVM(kernel="rbf")
