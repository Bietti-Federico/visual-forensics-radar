"""Persists a trained model bundle to disk: the global entity classifiers
plus each entity's own detectors/models (if fit) and readiness metadata.

Scope: this only saves plugin objects exactly as given — it does not fit
anything itself. Call `RetrainModelsUseCase`
(`application/model_training/retrain_models_use_case.py`) first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pdf_forensics.application.entity_identification.ports import EntityClassifierPlugin
from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.infrastructure.model_persistence.model_store import save_bundle


class SaveTrainedModelsUseCase:
    def execute(
        self,
        entity_classifiers: Sequence[EntityClassifierPlugin],
        per_entity: Mapping[str, EntityModelBundle],
        output_path: Path,
    ) -> None:
        save_bundle(
            output_path,
            {
                "entity_classifiers": {
                    classifier.classifier_id: classifier for classifier in entity_classifiers
                },
                "per_entity": dict(per_entity),
            },
        )
