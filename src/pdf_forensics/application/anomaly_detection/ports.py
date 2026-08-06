"""The port anomaly-detection use cases depend on; `plugins/anomaly_detection/` implements it.

Unlike `feature_extraction`'s and `rule_engine`'s ports, this one is
deliberately **stateful**: an ML detector must be `fit()` on a reference
batch before it can `score()` anything, and every implementation is expected
to raise clearly if `score()` is called first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore


class AnomalyDetectorPlugin(Protocol):
    detector_id: str

    def fit(self, feature_vectors: Sequence[Mapping[str, float]]) -> None:
        """Fit this detector on a reference batch of numeric feature vectors."""
        ...

    def score(self, feature_vector: Mapping[str, float]) -> AnomalyScore:
        """Score one feature vector. Raises if called before `fit()`."""
        ...
