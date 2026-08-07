"""Weights for combining the seven available risk components into one score.

Defaults are a documented starting point, not a calibrated model — there is
no labeled dataset large enough yet to empirically fit these (that's exactly
what Training Data Ingestion → ML Ensemble is for, going forward). Kept as an
explicit, constructor-injectable dataclass specifically so they can be
recalibrated later without changing `GenerateRiskReportUseCase`'s structure.

All seven of the platform brief's weighted inputs are represented here.
`entity_consistency` (Entity Identification, `application/entity_identification/`)
covers the brief's `Generator Confidence` — how confidently this document's
structure/producer matches one of the known real-world templates it's been
trained on. `signature_integrity` (Signature Verification,
`application/signature_verification/`) covers `Fingerprint Similarity` in
spirit, not literally — Module 3 still has no reference corpus of known-good
fingerprints to compare against, but a cryptographically verified embedded
signature is a much stronger per-document integrity signal than a
similarity search would have been anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

_WEIGHT_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class RiskWeights:
    rule_engine: float = 0.24
    ml_probability: float = 0.24
    anomaly_detection: float = 0.16
    structural: float = 0.08
    metadata: float = 0.08
    entity_consistency: float = 0.10
    signature_integrity: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.rule_engine
            + self.ml_probability
            + self.anomaly_detection
            + self.structural
            + self.metadata
            + self.entity_consistency
            + self.signature_integrity
        )
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"RiskWeights must sum to 1.0, got {total}.")

    def without_ml_probability(self) -> RiskWeights:
        """
        Zeroes `ml_probability` and rescales the other six proportionally so
        they still sum to 1.0 — for an entity ML Ensemble hasn't been fit for
        yet (not enough labeled genuine/confirmed-fraud examples). See
        `application/document_scoring/score_document_use_case.py`.
        """
        remainder = 1.0 - self.ml_probability
        if remainder <= 0.0:
            raise ValueError("Cannot rescale: ml_probability accounts for the entire weight.")
        factor = 1.0 / remainder
        return RiskWeights(
            rule_engine=self.rule_engine * factor,
            ml_probability=0.0,
            anomaly_detection=self.anomaly_detection * factor,
            structural=self.structural * factor,
            metadata=self.metadata * factor,
            entity_consistency=self.entity_consistency * factor,
            signature_integrity=self.signature_integrity * factor,
        )
