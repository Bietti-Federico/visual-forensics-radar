"""One detector's opinion of how anomalous a document's feature vector is.

`score` is standardized so higher always means more anomalous, across all
detectors — the underlying algorithms don't agree on that sign by default
(see each `plugins/anomaly_detection/*_detector.py` for its own conversion).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    detector_id: str
    score: float
    is_anomaly: bool
