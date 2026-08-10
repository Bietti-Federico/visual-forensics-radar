"""Combines Rule Engine, Anomaly Detection, ML Ensemble, Entity Identification,
Signature Verification, and structural/metadata signals into a single
weighted 0-100 risk score, packaged with Module 7's ExplanationReport.

All seven of the platform brief's weighted inputs are represented — see
`domain/risk/risk_weights.py`'s docstring for how `entity_consistency` and
`signature_integrity` map onto the brief's `Generator Confidence` and
`Fingerprint Similarity`.

The formulas below are a documented MVP starting point, not a calibrated
model — there is no labeled dataset large enough yet to empirically fit
these weights. `RiskWeights` is constructor-injectable specifically so this
can be recalibrated later without changing this use case's structure.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.explainability.generate_explanation_use_case import (
    GenerateExplanationUseCase,
)
from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.risk.risk_component_score import RiskComponentScore
from pdf_forensics.domain.risk.risk_report import RiskReport
from pdf_forensics.domain.risk.risk_weights import RiskWeights
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport
from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)

_RULE_SEVERITY_BASE_PROBABILITY = {
    AnomalySeverity.CRITICAL: 0.9,
    AnomalySeverity.WARNING: 0.5,
    AnomalySeverity.INFO: 0.15,
}
_STRUCTURAL_DENSITY_MULTIPLIER = 10.0
_METADATA_PRESENCE_FEATURES = (
    "metadata.has_title",
    "metadata.has_author",
    "metadata.has_creator",
    "metadata.has_producer",
    "metadata.has_creation_date",
    "metadata.has_mod_date",
)


class GenerateRiskReportUseCase:
    def __init__(self, weights: RiskWeights | None = None) -> None:
        self._weights = weights or RiskWeights()
        self._explain = GenerateExplanationUseCase()

    def execute(
        self,
        feature_set: FeatureSet,
        rule_report: RuleEvaluationReport,
        anomaly_report: AnomalyDetectionReport,
        ml_report: MlEnsembleReport,
        shap_explanations: Sequence[ShapExplanation],
        entity_report: EntityIdentificationReport,
        signature_report: SignatureVerificationReport,
    ) -> RiskReport:
        explanation = self._explain.execute(
            rule_report, anomaly_report, ml_report, shap_explanations
        )

        components = (
            RiskComponentScore(
                name="rule_engine",
                score=self._rule_engine_score(rule_report),
                weight=self._weights.rule_engine,
            ),
            RiskComponentScore(
                name="ml_probability",
                score=self._ml_probability_score(ml_report),
                weight=self._weights.ml_probability,
            ),
            RiskComponentScore(
                name="anomaly_detection",
                score=self._anomaly_detection_score(anomaly_report),
                weight=self._weights.anomaly_detection,
            ),
            RiskComponentScore(
                name="structural",
                score=self._structural_score(feature_set),
                weight=self._weights.structural,
            ),
            RiskComponentScore(
                name="metadata",
                score=self._metadata_score(feature_set),
                weight=self._weights.metadata,
            ),
            RiskComponentScore(
                name="entity_consistency",
                score=self._entity_consistency_score(entity_report),
                weight=self._weights.entity_consistency,
            ),
            RiskComponentScore(
                name="signature_integrity",
                score=self._signature_integrity_score(signature_report),
                weight=self._weights.signature_integrity,
            ),
        )

        weighted_sum = sum(component.score * component.weight for component in components)
        risk_score = round(max(0.0, min(1.0, weighted_sum)) * 100)

        return RiskReport(risk_score=risk_score, components=components, explanation=explanation)

    def _rule_engine_score(self, rule_report: RuleEvaluationReport) -> float:
        probability_of_no_risk = 1.0
        for finding in rule_report:
            base = _RULE_SEVERITY_BASE_PROBABILITY[finding.severity]
            probability_of_no_risk *= 1.0 - base
        return 1.0 - probability_of_no_risk

    def _anomaly_detection_score(self, anomaly_report: AnomalyDetectionReport) -> float:
        """
        Fraction of configured detectors that flagged the document, not a
        Noisy-OR with a flat per-detector weight. A flat weight (e.g. 0.7)
        made a single detector sitting exactly on its own decision boundary
        (a documented artifact with small per-entity training batches — see
        `docs/DOCUMENTACION.md`) contribute as much as several independent
        detectors actually agreeing, producing the same score for a
        genuinely anomalous document and one with one borderline flag.
        Agreement across more of the ensemble now scores higher than a
        single flag, without needing to special-case any one detector.
        """
        scores = list(anomaly_report)
        if not scores:
            return 0.0
        flagged = sum(1 for score in scores if score.is_anomaly)
        return flagged / len(scores)

    def _ml_probability_score(self, ml_report: MlEnsembleReport) -> float:
        probabilities = [prediction.probability for prediction in ml_report]
        if not probabilities:
            return 0.0
        return sum(probabilities) / len(probabilities)

    def _structural_score(self, feature_set: FeatureSet) -> float:
        anomaly_count_feature = feature_set.by_name("statistics.anomaly_count_total")
        object_count_feature = feature_set.by_name("general.object_count")
        if anomaly_count_feature is None or object_count_feature is None:
            return 0.0
        anomaly_count = anomaly_count_feature.value
        object_count = object_count_feature.value
        if not isinstance(anomaly_count, int) or not isinstance(object_count, int):
            return 0.0
        density = anomaly_count / max(1, object_count) * _STRUCTURAL_DENSITY_MULTIPLIER
        return min(1.0, density)

    def _metadata_score(self, feature_set: FeatureSet) -> float:
        has_info_dict = feature_set.by_name("metadata.has_info_dict")
        if has_info_dict is not None and has_info_dict.value is False:
            return 1.0

        present_features = [feature_set.by_name(name) for name in _METADATA_PRESENCE_FEATURES]
        known_features = [feature for feature in present_features if feature is not None]
        if not known_features:
            return 0.0
        missing_count = sum(1 for feature in known_features if feature.value is False)
        return missing_count / len(known_features)

    def _entity_consistency_score(self, entity_report: EntityIdentificationReport) -> float:
        """
        `1 - confidence`, not a true claimed-vs-actual mismatch check: this
        platform has no way yet to read which institution a document's own
        *visible content* claims to be from (Module 2 does structural/
        metadata features, not page text layout) — see `docs/DOCUMENTACION.md`.
        A low score here means "this document's structure/producer doesn't
        confidently match any of the real-world templates the classifier has
        seen," a genuine and useful signal on its own, just not the full
        claimed-vs-actual check the brief's `Generator Confidence` ultimately
        implies.
        """
        confidences = [prediction.confidence for prediction in entity_report]
        if not confidences:
            return 0.0
        return 1.0 - (sum(confidences) / len(confidences))

    def _signature_integrity_score(self, signature_report: SignatureVerificationReport) -> float:
        """
        `0.0` when no signature is present at all — the absence of a
        signature isn't itself suspicious (most documents this platform
        handles have none); `entity_template_mismatch`
        (`application/entity_invariants/check_entity_invariants_use_case.py`)
        already covers "this entity's genuine documents always have one and
        this doesn't."

        When a signature IS present, the worst finding across all of them
        wins: a broken digest (content changed after signing) is worse than
        a signature that doesn't cover the whole file (content appended
        after signing), which is worse than a malformed signature blob that
        still hashes correctly. Trust-chain validity is deliberately not
        part of this score — see the domain result type's docstring.
        """
        if not signature_report.results:
            return 0.0

        score = 0.0
        for result in signature_report:
            if not result.digest_intact:
                score = max(score, 1.0)
            elif result.coverage is not SignatureCoverage.ENTIRE_FILE:
                score = max(score, 0.7)
            elif not result.cryptographically_valid:
                score = max(score, 0.9)
        return score
