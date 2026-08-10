"""Scores one PDF through the full pipeline using an already-trained model bundle.

Single shared implementation for whatever previously called each module in
sequence by hand (the CLI scripts and, now, the API's `/verify` endpoint) —
see `application/model_training/retrain_models_use_case.py` for how the
bundle this depends on gets built.

Entirely a pure function of `(pdf_bytes, loaded bundle)`: no disk writes, no
mutation of the bundle it was constructed with. Safe to call from a
request handler with the bundle loaded once at process startup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pdf_forensics.application.anomaly_detection.detect_anomalies_use_case import (
    DetectAnomaliesUseCase,
)
from pdf_forensics.application.entity_identification.identify_entity_use_case import (
    IdentifyEntityUseCase,
)
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.entity_invariants.check_entity_invariants_use_case import (
    CheckEntityInvariantsUseCase,
)
from pdf_forensics.application.explainability.generate_explanation_use_case import (
    GenerateExplanationUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.fingerprinting.generate_fingerprint_use_case import (
    GenerateFingerprintUseCase,
)
from pdf_forensics.application.ml_ensemble.explain_prediction_use_case import (
    ExplainPredictionUseCase,
)
from pdf_forensics.application.ml_ensemble.predict_use_case import PredictUseCase
from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.risk_report.generate_risk_report_use_case import (
    GenerateRiskReportUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_rules_use_case import EvaluateRulesUseCase
from pdf_forensics.application.signature_verification.verify_signatures_use_case import (
    VerifySignaturesUseCase,
)
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.fingerprint.fingerprint import PdfFingerprint
from pdf_forensics.domain.risk.risk_report import RiskReport
from pdf_forensics.domain.risk.risk_weights import RiskWeights
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.rules import default_rules


@dataclass(frozen=True, slots=True)
class DocumentScoringResult:
    fingerprint: PdfFingerprint
    entity_report: EntityIdentificationReport
    signature_report: SignatureVerificationReport
    explanation: ExplanationReport
    risk_report: RiskReport


class ScoreDocumentUseCase:
    def __init__(
        self,
        entity_classifiers: Sequence[EntityClassifierPlugin],
        per_entity: Mapping[str, EntityModelBundle],
    ) -> None:
        self._entity_classifiers = tuple(entity_classifiers)
        self._per_entity = dict(per_entity)

    def execute(self, pdf_bytes: bytes) -> DocumentScoringResult:
        parse_pdf = ParsePdfUseCase()
        extract_features = FeatureExtractionUseCase(default_feature_extractors())

        document = parse_pdf.execute(pdf_bytes)
        feature_set = extract_features.execute(document)
        fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

        entity_report = IdentifyEntityUseCase(self._entity_classifiers).execute(feature_set)
        # Exactly one classifier is configured by design (see
        # plugins/entity_identification/__init__.py) — its top prediction
        # picks which entity's bundle to use.
        bundle = (
            self._per_entity.get(entity_report.predictions[0].predicted_entity)
            if entity_report.predictions
            else None
        )
        detectors = bundle.detectors if bundle is not None else ()
        models = bundle.models if bundle is not None else ()
        invariants = bundle.invariants if bundle is not None else ()
        ml_ensemble_ready = bundle.ml_ensemble_ready if bundle is not None else False

        anomaly_report = DetectAnomaliesUseCase(detectors).execute(feature_set)
        ml_report = PredictUseCase(models).execute(feature_set)
        shap_explanations = ExplainPredictionUseCase(models).execute(feature_set)
        signature_report = VerifySignaturesUseCase().execute(pdf_bytes)

        plain_rule_report = EvaluateRulesUseCase(default_rules()).execute(feature_set)
        invariant_finding = CheckEntityInvariantsUseCase(invariants).execute(
            feature_set, entity_report
        )
        rule_report = RuleEvaluationReport(
            findings=list(plain_rule_report) + ([invariant_finding] if invariant_finding else [])
        )

        weights = RiskWeights() if ml_ensemble_ready else RiskWeights().without_ml_probability()
        explanation = GenerateExplanationUseCase().execute(
            rule_report, anomaly_report, ml_report, shap_explanations
        )
        risk_report = GenerateRiskReportUseCase(weights=weights).execute(
            feature_set,
            rule_report,
            anomaly_report,
            ml_report,
            shap_explanations,
            entity_report,
            signature_report,
        )

        return DocumentScoringResult(
            fingerprint=fingerprint,
            entity_report=entity_report,
            signature_report=signature_report,
            explanation=explanation,
            risk_report=risk_report,
        )
