"""One labeled training row (and the bookkeeping for rows that couldn't be built)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.training_data.benchmark_sample_ref import BenchmarkSampleRef


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """A benchmark-generated document, re-featurized in the validator's own feature space."""

    id: str
    features: FeatureSet
    source: BenchmarkSampleRef

    def to_row(self) -> dict[str, Any]:
        """Flatten to one ML-ready row: every extracted feature plus the label columns."""
        return {
            "id": self.id,
            **self.features.to_dict(),
            "seed": self.source.seed,
            "original_id": self.source.original_id,
            "parent_id": self.source.parent_id,
            "generator": self.source.generator,
            "pipeline": "|".join(self.source.pipeline),
            "pipeline_depth": self.source.pipeline_depth,
            "is_original": self.source.is_original,
            "binary_class": self.source.binary_class,
            "multiclass_label": self.source.multiclass_label,
        }


@dataclass(frozen=True, slots=True)
class SkippedSample:
    """A benchmark-generated document that couldn't be re-parsed/re-featurized, and why."""

    id: str
    path: Path
    reason: str


@dataclass(slots=True)
class TrainingDataset:
    samples: list[TrainingSample] = field(default_factory=list)
    skipped: list[SkippedSample] = field(default_factory=list)

    def to_rows(self) -> list[dict[str, Any]]:
        return [sample.to_row() for sample in self.samples]
