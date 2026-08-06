"""Weights for combining the six available risk components into one score.

Defaults are a documented starting point, not a calibrated model — there is
no labeled dataset large enough yet to empirically fit these (that's exactly
what Training Data Ingestion → ML Ensemble is for, going forward). Kept as an
explicit, constructor-injectable dataclass specifically so they can be
recalibrated later without changing `GenerateRiskReportUseCase`'s structure.

Six of the platform brief's seven weighted inputs are represented here.
`entity_consistency` (Entity Identification, `application/entity_identification/`)
covers the brief's `Generator Confidence` — how confidently this document's
structure/producer matches one of the known real-world templates it's been
trained on. `Fingerprint Similarity` (needs a reference corpus of known-good
fingerprints to compare against, which doesn't exist — Module 3 only
computes one document's fingerprint) is still genuinely unavailable, not
weighted at zero silently.
"""

from __future__ import annotations

from dataclasses import dataclass

_WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RiskWeights:
    rule_engine: float = 0.27
    ml_probability: float = 0.27
    anomaly_detection: float = 0.18
    structural: float = 0.09
    metadata: float = 0.09
    entity_consistency: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.rule_engine
            + self.ml_probability
            + self.anomaly_detection
            + self.structural
            + self.metadata
            + self.entity_consistency
        )
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"RiskWeights must sum to 1.0, got {total}.")
