"""Persists already-fitted Anomaly Detection / ML Ensemble / Entity Identification
plugin instances to disk.

Scope: this only saves plugin objects exactly as given — it does not fit
anything itself. Call `FitAnomalyDetectorsUseCase`/`TrainModelsUseCase`/
`TrainEntityClassifierUseCase` first. `entity_classifiers` defaults to empty
so existing callers that only ever trained detectors/models keep working.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.infrastructure.model_persistence.model_store import save_bundle


class SaveTrainedModelsUseCase:
    def execute(
        self,
        detectors: Sequence[AnomalyDetectorPlugin],
        models: Sequence[SupervisedModelPlugin],
        output_path: Path,
        entity_classifiers: Sequence[EntityClassifierPlugin] = (),
    ) -> None:
        save_bundle(
            output_path,
            {
                "detectors": {detector.detector_id: detector for detector in detectors},
                "models": {model.model_id: model for model in models},
                "entity_classifiers": {
                    classifier.classifier_id: classifier for classifier in entity_classifiers
                },
            },
        )
