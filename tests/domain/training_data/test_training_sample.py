from pathlib import Path

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.training_data.benchmark_sample_ref import BenchmarkSampleRef
from pdf_forensics.domain.training_data.training_sample import (
    SkippedSample,
    TrainingDataset,
    TrainingSample,
)


def _ref(**overrides: object) -> BenchmarkSampleRef:
    defaults: dict[str, object] = {
        "id": "doc-1",
        "path": Path("/data/doc-1.pdf"),
        "seed": 42,
        "original_id": "doc-1",
        "parent_id": None,
        "generator": "receipt_generator",
        "pipeline": ("pikepdf_metadata_removal",),
        "pipeline_depth": 1,
        "is_original": False,
        "binary_class": "transformed",
        "multiclass_label": "pikepdf_metadata_removal",
    }
    defaults.update(overrides)
    return BenchmarkSampleRef(**defaults)  # type: ignore[arg-type]


def _feature_set() -> FeatureSet:
    return FeatureSet(
        features=[
            Feature(
                name="general.object_count",
                value=3,
                value_type=FeatureType.INTEGER,
                description="d",
                source="s",
                confidence=1.0,
                category=FeatureCategory.GENERAL,
                schema_version="1.0.0",
            )
        ]
    )


def test_to_row_flattens_features_and_labels() -> None:
    sample = TrainingSample(id="doc-1", features=_feature_set(), source=_ref())
    row = sample.to_row()

    assert row["id"] == "doc-1"
    assert row["general.object_count"] == 3
    assert row["seed"] == 42
    assert row["pipeline"] == "pikepdf_metadata_removal"
    assert row["is_original"] is False
    assert row["binary_class"] == "transformed"
    assert row["multiclass_label"] == "pikepdf_metadata_removal"


def test_training_dataset_to_rows() -> None:
    sample = TrainingSample(id="doc-1", features=_feature_set(), source=_ref())
    dataset = TrainingDataset(samples=[sample])
    assert dataset.to_rows() == [sample.to_row()]


def test_empty_training_dataset_defaults() -> None:
    dataset = TrainingDataset()
    assert dataset.samples == []
    assert dataset.skipped == []
    assert dataset.to_rows() == []


def test_skipped_sample_fields() -> None:
    skipped = SkippedSample(id="doc-2", path=Path("/data/doc-2.pdf"), reason="not a pdf")
    assert skipped.id == "doc-2"
    assert skipped.reason == "not a pdf"
