"""The complete set of anomaly scores produced by one detection run."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore


@dataclass(slots=True)
class AnomalyDetectionReport:
    scores: list[AnomalyScore] = field(default_factory=list)

    def by_detector(self, detector_id: str) -> AnomalyScore | None:
        for score in self.scores:
            if score.detector_id == detector_id:
                return score
        return None

    def __len__(self) -> int:
        return len(self.scores)

    def __iter__(self) -> Iterator[AnomalyScore]:
        return iter(self.scores)
