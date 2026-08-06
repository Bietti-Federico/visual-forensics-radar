"""Restores plugin instances previously saved by `SaveTrainedModelsUseCase`.

Returned instances are already fitted — they can be passed straight into
`DetectAnomaliesUseCase`/`PredictUseCase`/`ExplainPredictionUseCase` without
calling `fit()`/`TrainModelsUseCase` again.
"""

from __future__ import annotations

from pathlib import Path

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.infrastructure.model_persistence.model_store import load_bundle


class LoadTrainedModelsUseCase:
    def execute(self, input_path: Path) -> tuple[
        tuple[AnomalyDetectorPlugin, ...],
        tuple[SupervisedModelPlugin, ...],
        tuple[EntityClassifierPlugin, ...],
    ]:
        bundle = load_bundle(input_path)
        detectors = tuple(bundle["detectors"].values())
        models = tuple(bundle["models"].values())
        entity_classifiers = tuple(bundle.get("entity_classifiers", {}).values())
        return detectors, models, entity_classifiers
