"""End-to-end manual sanity check: run every module (1 through 8) on one PDF.

Usage:
    poetry run python scripts/run_full_pipeline.py [path/to/file.pdf]

Without an argument, parses a small built-in synthetic PDF instead.

Anomaly Detection (Module 5) and ML Ensemble (Module 6) both need to be fit
on a reference batch before they can score/predict anything. This script
builds a small SYNTHETIC reference batch inline (a few single-revision
"normal" documents and a few many-revision "unusual" ones) purely so this
demo is self-contained and runnable with zero setup — it is not a real
training corpus. For actual use, fit these on a real dataset produced by the
Training Data Ingestion module (`application/training_data/`).
"""

# ruff: noqa: E402  (sys.path must be extended before the tests.fixtures import below)
from __future__ import annotations

import sys
from pathlib import Path

# Running this script directly (not via pytest) doesn't get the repo root on
# sys.path automatically, but it's needed to reuse tests/fixtures/pdf_builder.py
# for the demo reference batch below (see module docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fixtures.pdf_builder import PdfBuilder

from pdf_forensics.application.anomaly_detection.detect_anomalies_use_case import (
    DetectAnomaliesUseCase,
)
from pdf_forensics.application.anomaly_detection.fit_anomaly_detectors_use_case import (
    FitAnomalyDetectorsUseCase,
)
from pdf_forensics.application.entity_identification.identify_entity_use_case import (
    IdentifyEntityUseCase,
)
from pdf_forensics.application.entity_identification.train_entity_classifier_use_case import (
    TrainEntityClassifierUseCase,
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
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.risk_report.generate_risk_report_use_case import (
    GenerateRiskReportUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_rules_use_case import EvaluateRulesUseCase
from pdf_forensics.application.signature_verification.verify_signatures_use_case import (
    VerifySignaturesUseCase,
)
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models
from pdf_forensics.plugins.rules import default_rules


def _synthetic_pdf_bytes() -> bytes:
    builder = PdfBuilder()
    off_info = builder.add_object(
        4, 0, "<< /Title (Sample) /Producer (Acme PDF) /Creator (Acme Writer) >>"
    )
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off3 = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R >>")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, off3, 0, "n"),
            (4, off_info, 0, "n"),
        ],
        size=5,
        root_ref="1 0 R",
        extra_trailer="/Info 4 0 R",
    )
    return builder.build()


def _normal_reference_document(variant: int) -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder.build()


def _unusual_reference_document(variant: int) -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    prev = builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    for _ in range(5):
        off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R /Touched true >>")
        prev = builder.add_classic_xref_and_trailer(
            [(1, off1, 0, "n")], size=3, root_ref="1 0 R", prev=prev
        )
    return builder.build()


def main() -> None:
    parse_pdf = ParsePdfUseCase()
    extract_features = FeatureExtractionUseCase(default_feature_extractors())

    # --- Build a small synthetic reference batch (demo only, see module docstring) ---
    normal_feature_sets = [
        extract_features.execute(parse_pdf.execute(_normal_reference_document(i)))
        for i in range(10)
    ]
    unusual_feature_sets = [
        extract_features.execute(parse_pdf.execute(_unusual_reference_document(i)))
        for i in range(10)
    ]
    reference_feature_sets = normal_feature_sets + unusual_feature_sets
    reference_labels = [True] * len(normal_feature_sets) + [False] * len(unusual_feature_sets)

    detectors = default_detectors()
    FitAnomalyDetectorsUseCase(detectors).execute(reference_feature_sets)

    models = default_models()
    TrainModelsUseCase(models).execute(reference_feature_sets, reference_labels)

    # Single-class by construction (this demo has no multi-entity data) — the
    # entity classifier will trivially predict this one label with 100%
    # confidence for everything. Exercises the pipeline wiring, not real
    # entity-identification signal; see scripts/train_and_score_real_documents.py
    # for that.
    entity_classifiers = default_entity_classifiers()
    TrainEntityClassifierUseCase(entity_classifiers).execute(
        reference_feature_sets, ["SYNTHETIC_REFERENCE"] * len(reference_feature_sets)
    )

    # --- Parse the target document (Module 1) ---
    if len(sys.argv) > 1:
        data = Path(sys.argv[1]).read_bytes()
    else:
        data = _synthetic_pdf_bytes()
    document = parse_pdf.execute(data)

    # --- Modules 2-6 ---
    feature_set = extract_features.execute(document)
    signature_report = VerifySignaturesUseCase().execute(data)
    fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)
    anomaly_report = DetectAnomaliesUseCase(detectors).execute(feature_set)
    ml_report = PredictUseCase(models).execute(feature_set)
    shap_explanations = ExplainPredictionUseCase(models).execute(feature_set)
    entity_report = IdentifyEntityUseCase(entity_classifiers).execute(feature_set)

    # No per-entity learned invariants here — this demo's reference batch is
    # synthetic and single-label (see module docstring), not real per-entity
    # training data (see `application/entity_invariants/` for that).
    rule_report = EvaluateRulesUseCase(default_rules()).execute(feature_set)

    # --- Module 7 (shown standalone, also embedded in Module 8's output below) ---
    explanation = GenerateExplanationUseCase().execute(
        rule_report, anomaly_report, ml_report, shap_explanations
    )

    # --- Module 8: the final deliverable ---
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
