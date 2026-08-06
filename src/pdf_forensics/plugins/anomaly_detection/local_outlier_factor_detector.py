"""Local Outlier Factor anomaly detector, in novelty mode (sklearn.neighbors.LocalOutlierFactor).

`novelty=True` is required to score data points not seen during `fit()` at
all (the default LOF mode only supports scoring the training set itself) —
this is also exactly the platform brief's "Novelty Detection" requirement.
`n_neighbors` is capped to the fit batch size: sklearn's default (20) raises
on any fit batch smaller than that, which real (especially test) batches
often are.
"""

from __future__ import annotations

from typing import Any

from sklearn.neighbors import LocalOutlierFactor

from pdf_forensics.plugins.anomaly_detection._sklearn_base import SklearnAnomalyDetector


class LocalOutlierFactorDetector(SklearnAnomalyDetector):
    detector_id = "local_outlier_factor"

    def _build_estimator(self, n_samples: int) -> Any:
        n_neighbors = max(1, min(20, n_samples - 1))
        return LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)
