"""
Fits Anomaly Detection (Module 5) and ML Ensemble (Module 6) on a REAL
benchmark run (real user-provided PDFs + realistic transformation variants,
built by pdf-forensics-benchmark's `examples/build_real_documents_dataset.py`)
instead of the throwaway synthetic grid `run_full_pipeline.py` uses, then
scores every sample in that same run through the full Module 1-8 pipeline —
plus an optional extra target file.

Usage:
    poetry run python scripts/train_and_score_real_documents.py <benchmark_output_dir> \
        [--target-pdf PATH] [--real-weight FLOAT]

Caveat this script does not hide: a handful of real documents (each with a
handful of transformation variants) is a very small training set — enough to
exercise every module end-to-end on real data, not enough to calibrate a
production model. Genuinely real documents (and their direct transformations)
are up-weighted relative to synthetic field-substituted variants — see
scripts/_training_weights.py — but that's a mitigation, not a substitute for
more real documents. Model persistence lives in
`application/model_persistence/` (see scripts/train_and_save_models.py) —
this script always refits from scratch, by design, so it stays a simple,
self-contained sanity check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _entity_labels import compute_entity_labels
from _training_weights import compute_sample_weight

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
from pdf_forensics.application.rule_engine.evaluate_entity_aware_rules_use_case import (
    EvaluateEntityAwareRulesUseCase,
)
from pdf_forensics.application.rule_engine.evaluate_rules_use_case import EvaluateRulesUseCase
from pdf_forensics.application.training_data.build_training_dataset_use_case import (
    BuildTrainingDatasetUseCase,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models
from pdf_forensics.plugins.rules import default_entity_aware_rules, default_rules


def _print_risk_report(
    label: str, feature_set: FeatureSet, detectors, models, entity_classifiers
) -> None:
    anomaly_report = DetectAnomaliesUseCase(detectors).execute(feature_set)
    ml_report = PredictUseCase(models).execute(feature_set)
    shap_explanations = ExplainPredictionUseCase(models).execute(feature_set)
    entity_report = IdentifyEntityUseCase(entity_classifiers).execute(feature_set)

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
        feature_set, rule_report, anomaly_report, ml_report, shap_explanations, entity_report
    )

    print(f"\n=== {label} ===")
    for prediction in entity_report:
        print(
            f"  entity[{prediction.classifier_id}]: {prediction.predicted_entity} "
            f"(confidence={prediction.confidence:.0%})"
        )
    print(f"Risk Score: {risk_report.risk_score}/100")
    for component in risk_report.components:
        print(f"  {component.name}: score={component.score:.3f} weight={component.weight}")
    for reason in explanation.reasons:
        print(f"  - {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_output_dir", type=Path)
    parser.add_argument("--target-pdf", type=Path, default=None)
    parser.add_argument(
        "--real-weight",
        type=float,
        default=8.0,
        help="How many times more a genuinely real document counts vs. a synthetic variant.",
    )
    args = parser.parse_args()

    benchmark_output_dir = args.benchmark_output_dir
    target_pdf = args.target_pdf
    real_weight = args.real_weight

    parse_pdf = ParsePdfUseCase()
    extract_features = FeatureExtractionUseCase(default_feature_extractors())

    # --- Build the training dataset from the real benchmark run ---
    dataset = BuildTrainingDatasetUseCase(
        parse_pdf_use_case=parse_pdf,
        feature_extraction_use_case=extract_features,
    ).execute(benchmark_output_dir)

    print(f"Training samples: {len(dataset.samples)}")
    print(f"Skipped: {len(dataset.skipped)}")
    for skipped in dataset.skipped:
        print(f"  {skipped.id} ({skipped.path}): {skipped.reason}")

    feature_sets = [sample.features for sample in dataset.samples]
    # TrainModelsUseCase (and SklearnClassifierPlugin.fit internally) expect
    # `labels` as `is_original` — NOT already flipped to "is_manipulated".
    # See train_models_use_case.py's own module docstring.
    labels = [sample.source.is_original for sample in dataset.samples]
    sample_weight = compute_sample_weight(dataset, real_weight)
    real_count = sum(1 for w in sample_weight if w != 1.0)
    print(f"Real-document samples up-weighted to {real_weight}x: {real_count}/{len(sample_weight)}")

    detectors = default_detectors()
    FitAnomalyDetectorsUseCase(detectors).execute(feature_sets, sample_weight=sample_weight)

    models = default_models()
    TrainModelsUseCase(models).execute(feature_sets, labels, sample_weight=sample_weight)

    entity_classifiers = default_entity_classifiers()
    entity_labels = compute_entity_labels(dataset)
    known = [
        (fs, label, weight)
        for fs, label, weight in zip(feature_sets, entity_labels, sample_weight, strict=True)
        if label is not None
    ]
    unknown_count = len(feature_sets) - len(known)
    if unknown_count:
        print(f"Entity identification: skipping {unknown_count} samples with no known entity label")
    TrainEntityClassifierUseCase(entity_classifiers).execute(
        [fs for fs, _, _ in known],
        [label for _, label, _ in known],
        sample_weight=[weight for _, _, weight in known],
    )

    # --- Score every sample from the training run itself ---
    for sample in dataset.samples:
        label = "original" if sample.source.is_original else sample.source.multiclass_label
        _print_risk_report(
            f"{sample.id} ({label})", sample.features, detectors, models, entity_classifiers
        )

    # --- Score an optional extra target file, fully held out of training ---
    if target_pdf is not None:
        document = parse_pdf.execute(target_pdf.read_bytes())
        feature_set = extract_features.execute(document)
        fingerprint = GenerateFingerprintUseCase().execute(document, feature_set)
        print(f"\nFingerprint ({target_pdf.name}): {fingerprint.to_dict()}")
        _print_risk_report(
            f"TARGET: {target_pdf.name}", feature_set, detectors, models, entity_classifiers
        )


if __name__ == "__main__":
    main()
