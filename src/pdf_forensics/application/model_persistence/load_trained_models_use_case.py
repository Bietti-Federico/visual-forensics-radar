"""Restores a trained model bundle previously saved by `SaveTrainedModelsUseCase`.

Returned plugin instances are already fitted — pass them straight into
`IdentifyEntityUseCase`/`DetectAnomaliesUseCase`/`PredictUseCase`/
`ExplainPredictionUseCase` without fitting/training again.
"""

from __future__ import annotations

from pathlib import Path

from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.infrastructure.model_persistence.model_store import load_bundle


class LoadTrainedModelsUseCase:
    def execute(
        self, input_path: Path
    ) -> tuple[tuple[EntityClassifierPlugin, ...], dict[str, EntityModelBundle]]:
        bundle = load_bundle(input_path)
        entity_classifiers = tuple(bundle["entity_classifiers"].values())
        per_entity: dict[str, EntityModelBundle] = bundle.get("per_entity", {})
        return entity_classifiers, per_entity
