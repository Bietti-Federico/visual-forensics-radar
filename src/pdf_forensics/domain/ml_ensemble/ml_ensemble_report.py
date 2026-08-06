"""The complete set of predictions produced by one ensemble run."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction


@dataclass(slots=True)
class MlEnsembleReport:
    predictions: list[ModelPrediction] = field(default_factory=list)

    def by_model(self, model_id: str) -> ModelPrediction | None:
        for prediction in self.predictions:
            if prediction.model_id == model_id:
                return prediction
        return None

    def __len__(self) -> int:
        return len(self.predictions)

    def __iter__(self) -> Iterator[ModelPrediction]:
        return iter(self.predictions)
