"""
Fits Anomaly Detection (Module 5), ML Ensemble (Module 6), and Entity
Identification on a benchmark run (see
scripts/train_and_score_real_documents.py for the same training step) and
persists the fitted detectors/models/classifiers to disk, so scoring a new
document later (scripts/score_document.py) doesn't require retraining.

Usage:
    poetry run python scripts/train_and_save_models.py \
        <benchmark_output_dir> <model_store_path> [real_weight]

`real_weight` (default 16.0) is how many times more each genuinely real
document (and its direct transformations) counts relative to a synthetic
field-substituted variant — see scripts/_training_weights.py. Chosen via a
leave-one-real-document-out sweep: ML Ensemble separation between held-out
real originals and their transforms keeps improving up to at least 32x
(0.937 at 8x -> 0.948 at 16x -> 0.951 at 32x, real_probability/mean), while
Entity Identification confidence on the two smallest-n (Jujuy) documents
starts dropping past 16x — 16x captures most of the ML Ensemble gain without
that cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

from _entity_labels import compute_entity_labels
from _training_weights import compute_sample_weight

from pdf_forensics.application.anomaly_detection.fit_anomaly_detectors_use_case import (
    FitAnomalyDetectorsUseCase,
)
from pdf_forensics.application.entity_identification.train_entity_classifier_use_case import (
    TrainEntityClassifierUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.application.model_persistence.save_trained_models_use_case import (
    SaveTrainedModelsUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.training_data.build_training_dataset_use_case import (
    BuildTrainingDatasetUseCase,
)
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models


def main() -> None:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)

    benchmark_output_dir = Path(sys.argv[1])
    model_store_path = Path(sys.argv[2])
    real_weight = float(sys.argv[3]) if len(sys.argv) == 4 else 16.0

    dataset = BuildTrainingDatasetUseCase(
        parse_pdf_use_case=ParsePdfUseCase(),
        feature_extraction_use_case=FeatureExtractionUseCase(default_feature_extractors()),
    ).execute(benchmark_output_dir)

    print(f"Training samples: {len(dataset.samples)}")
    print(f"Skipped: {len(dataset.skipped)}")
    for skipped in dataset.skipped:
        print(f"  {skipped.id} ({skipped.path}): {skipped.reason}")

    feature_sets = [sample.features for sample in dataset.samples]
    # TrainModelsUseCase/SklearnClassifierPlugin.fit expect `labels` as
    # `is_original` (see train_models_use_case.py's own module docstring).
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
    known_feature_sets = [fs for fs, _, _ in known]
    known_entity_labels = [label for _, label, _ in known]
    known_weight = [weight for _, _, weight in known]
    TrainEntityClassifierUseCase(entity_classifiers).execute(
        known_feature_sets, known_entity_labels, sample_weight=known_weight
    )

    SaveTrainedModelsUseCase().execute(detectors, models, model_store_path, entity_classifiers)
    print(f"Saved trained detectors/models/entity-classifiers to {model_store_path}")


if __name__ == "__main__":
    main()
