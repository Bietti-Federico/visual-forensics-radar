"""Refits everything from the production training corpus and persists the result.

Orchestrates: `BuildTrainingCorpusUseCase` → per entity, fit Anomaly
Detection if enough genuine examples exist, fit ML Ensemble if enough
genuine *and* confirmed-fraud examples exist → one global entity classifier
across every entity seen → `SaveTrainedModelsUseCase`.

Thresholds are named constants, not configuration — deliberately not
over-engineered into env vars/a config file until there's a real need to
tune them per deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf_forensics.application.anomaly_detection.fit_anomaly_detectors_use_case import (
    FitAnomalyDetectorsUseCase,
)
from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.entity_identification.train_entity_classifier_use_case import (
    TrainEntityClassifierUseCase,
)
from pdf_forensics.application.entity_invariants.fit_entity_invariants_use_case import (
    FitEntityInvariantsUseCase,
)
from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.application.model_persistence.save_trained_models_use_case import (
    SaveTrainedModelsUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.training_corpus.build_training_corpus_use_case import (
    BuildTrainingCorpusUseCase,
)
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.features import default_feature_extractors
from pdf_forensics.plugins.ml_ensemble import default_models

#: Below this many genuine documents for an entity, even fitting Isolation
#: Forest/OneClassSVM/LOF/the autoencoder isn't meaningful.
ANOMALY_MIN_GENUINE_PER_ENTITY = 2
#: Both thresholds must be met before an entity's ML Ensemble activates.
ML_ENSEMBLE_MIN_GENUINE_PER_ENTITY = 5
ML_ENSEMBLE_MIN_CONFIRMED_FRAUD_PER_ENTITY = 3
#: "Constant across 2-3 documents" is a coincidence, not a template — this
#: is deliberately higher than `ANOMALY_MIN_GENUINE_PER_ENTITY` since a
#: learned invariant is a hard yes/no gate, not a statistical distance.
TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY = 4


@dataclass(frozen=True, slots=True)
class EntityRetrainSummary:
    entity: str
    genuine_count: int
    confirmed_fraud_count: int
    anomaly_detection_fitted: bool
    ml_ensemble_ready: bool
    invariant_count: int


@dataclass(frozen=True, slots=True)
class RetrainSummary:
    total_entries: int
    skipped_count: int
    per_entity: tuple[EntityRetrainSummary, ...]


class RetrainModelsUseCase:
    def __init__(self, model_store_path: Path) -> None:
        self._model_store_path = model_store_path

    def execute(self, corpus_dir: Path) -> RetrainSummary:
        corpus = BuildTrainingCorpusUseCase(
            parse_pdf_use_case=ParsePdfUseCase(),
            feature_extraction_use_case=FeatureExtractionUseCase(default_feature_extractors()),
        ).execute(corpus_dir)

        entity_bundles: dict[str, EntityModelBundle] = {}
        entity_summaries: list[EntityRetrainSummary] = []
        all_feature_sets: list[FeatureSet] = []
        all_entity_labels: list[str] = []

        for entity, entries in corpus.by_entity().items():
            genuine = [entry for entry in entries if entry.is_genuine]
            fraud = [entry for entry in entries if not entry.is_genuine]

            all_feature_sets.extend(entry.features for entry in entries)
            all_entity_labels.extend([entity] * len(entries))

            detectors: tuple[AnomalyDetectorPlugin, ...] = ()
            if len(genuine) >= ANOMALY_MIN_GENUINE_PER_ENTITY:
                detectors = default_detectors()
                FitAnomalyDetectorsUseCase(detectors).execute([entry.features for entry in genuine])

            ml_ensemble_ready = (
                len(genuine) >= ML_ENSEMBLE_MIN_GENUINE_PER_ENTITY
                and len(fraud) >= ML_ENSEMBLE_MIN_CONFIRMED_FRAUD_PER_ENTITY
            )
            models: tuple[SupervisedModelPlugin, ...] = ()
            if ml_ensemble_ready:
                models = default_models()
                feature_sets = [entry.features for entry in genuine] + [
                    entry.features for entry in fraud
                ]
                labels = [True] * len(genuine) + [False] * len(fraud)
                TrainModelsUseCase(models).execute(feature_sets, labels)

            invariants: tuple[LearnedInvariant, ...] = ()
            if len(genuine) >= TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY:
                invariants = FitEntityInvariantsUseCase().execute(
                    [entry.features for entry in genuine]
                )

            entity_bundles[entity] = EntityModelBundle(
                detectors=tuple(detectors),
                models=tuple(models),
                invariants=invariants,
                genuine_count=len(genuine),
                confirmed_fraud_count=len(fraud),
                ml_ensemble_ready=ml_ensemble_ready,
            )
            entity_summaries.append(
                EntityRetrainSummary(
                    entity=entity,
                    genuine_count=len(genuine),
                    confirmed_fraud_count=len(fraud),
                    anomaly_detection_fitted=bool(detectors),
                    invariant_count=len(invariants),
                    ml_ensemble_ready=ml_ensemble_ready,
                )
            )

        entity_classifiers: tuple[EntityClassifierPlugin, ...] = ()
        if corpus.entries:
            entity_classifiers = default_entity_classifiers()
            TrainEntityClassifierUseCase(entity_classifiers).execute(
                all_feature_sets, all_entity_labels
            )

        SaveTrainedModelsUseCase().execute(
            entity_classifiers, entity_bundles, self._model_store_path
        )

        return RetrainSummary(
            total_entries=len(corpus.entries),
            skipped_count=len(corpus.skipped),
            per_entity=tuple(entity_summaries),
        )
