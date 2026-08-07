"""
Scores one PDF through the full Module 1-8 pipeline using PREVIOUSLY TRAINED
and persisted Anomaly Detection / ML Ensemble models (see
scripts/train_and_save_models.py) — no retraining, instant scoring.

Usage:
    poetry run python scripts/score_document.py <model_store_path> <target_pdf>
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdf_forensics.application.anomaly_detection.detect_anomalies_use_case import (
    DetectAnomaliesUseCase,
)
from pdf_forensics.application.entity_identification.identify_entity_use_case import (
    IdentifyEntityUseCase,
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
from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.risk_report.generate_risk_report_use_case import (
    GenerateRiskReportUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_entity_aware_rules_use_case import (
    EvaluateEntityAwareRulesUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_rules_use_case import EvaluateRulesUseCase
from pdf_forensics.application.signature_verification.verify_signatures_use_case import (
    VerifySignaturesUseCase,
)
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.rules import default_entity_aware_rules, default_rules


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_store_path = Path(sys.argv[1])
    target_pdf = Path(sys.argv[2])

    detectors, models, entity_classifiers = LoadTrainedModelsUseCase().execute(model_store_path)

    parse_pdf = ParsePdfUseCase()
    extract_features = FeatureExtractionUseCase(default_feature_extractors())

    pdf_bytes = target_pdf.read_bytes()
    document = parse_pdf.execute(pdf_bytes)
    feature_set = extract_features.execute(document)
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)

    anomaly_report = DetectAnomaliesUseCase(detectors).execute(feature_set)
    ml_report = PredictUseCase(models).execute(feature_set)
    shap_explanations = ExplainPredictionUseCase(models).execute(feature_set)
    entity_report = IdentifyEntityUseCase(entity_classifiers).execute(feature_set)
    signature_report = VerifySignaturesUseCase().execute(pdf_bytes)

    plain_rule_report = EvaluateRulesUseCase(default_rules()).execute(feature_set)
    entity_aware_rule_report = EvaluateEntityAwareRulesUseCase(
        default_entity_aware_rules()
    ).execute(feature_set, entity_report)
    rule_report = RuleEvaluationReport(
        findings=list(plain_rule_report) + list(entity_aware_rule_report)
    )

    explanation = GenerateExplanationUseCase().execute(
        rule_report, anomaly_report, ml_report, shap_explanations
    )
    risk_report = GenerateRiskReportUseCase().execute(
        feature_set,
        rule_report,
        anomaly_report,
        ml_report,
        shap_explanations,
        entity_report,
        signature_report,
    )

    print(f"Fingerprint: {fingerprint.to_dict()}")
    print()
    print("Entity identification:")
    for prediction in entity_report:
        print(
            f"  [{prediction.classifier_id}] {prediction.predicted_entity} "
            f"(confidence={prediction.confidence:.0%}) — {prediction.probabilities}"
        )
    print()
    print("Signature verification:")
    if not signature_report:
        print("  No embedded signature found.")
    for result in signature_report:
        print(
            f"  [{result.field_name}] intact={result.digest_intact} "
            f"valid={result.cryptographically_valid} coverage={result.coverage.value} "
            f"signer={result.signer_subject!r} signed_at={result.signing_time}"
        )
    print()
    print(f"Risk Score: {risk_report.risk_score}/100")
    print("Components:")
    for component in risk_report.components:
        print(f"  {component.name}: score={component.score:.3f} weight={component.weight}")
    print("Reasons:")
    for reason in explanation.reasons:
        print(f"  - {reason}")
    print("Top SHAP features:")
    for feature in explanation.top_features:
        print(f"  [{feature.model_id}] {feature.feature_name}: {feature.shap_value:+.4f}")


if __name__ == "__main__":
    main()
