import csv
from pathlib import Path

from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.training_data.build_training_dataset_use_case import (
    BuildTrainingDatasetUseCase,
)
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder

_LABELS_FIELDS = [
    "id",
    "seed",
    "original_id",
    "parent_id",
    "generator",
    "pipeline",
    "transformation_count",
    "pipeline_depth",
    "is_original",
    "binary_class",
    "multiclass_label",
]
_HASHES_FIELDS = ["id", "path", "sha256", "sha1", "md5", "blake2b"]


def _label_row(sample_id: str, **overrides: str) -> dict:
    row = {
        "id": sample_id,
        "seed": "1",
        "original_id": sample_id,
        "parent_id": "",
        "generator": "receipt_generator",
        "pipeline": "",
        "transformation_count": "0",
        "pipeline_depth": "0",
        "is_original": "True",
        "binary_class": "original",
        "multiclass_label": "original",
    }
    row.update(overrides)
    return row


def _write_benchmark_output(tmp_path: Path, label_rows: list[dict], hash_rows: list[dict]) -> Path:
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LABELS_FIELDS)
        writer.writeheader()
        writer.writerows(label_rows)
    with (tmp_path / "hashes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HASHES_FIELDS)
        writer.writeheader()
        writer.writerows(hash_rows)
    return tmp_path


def _valid_pdf_bytes() -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder.build()


def _use_case() -> BuildTrainingDatasetUseCase:
    return BuildTrainingDatasetUseCase(
        parse_pdf_use_case=ParsePdfUseCase(),
        feature_extraction_use_case=FeatureExtractionUseCase(default_feature_extractors()),
    )


def test_successful_sample_is_featurized(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc-1.pdf"
    pdf_path.write_bytes(_valid_pdf_bytes())
    _write_benchmark_output(
        tmp_path,
        [_label_row("doc-1")],
        [
            {
                "id": "doc-1",
                "path": str(pdf_path),
                "sha256": "a",
                "sha1": "b",
                "md5": "c",
                "blake2b": "d",
            }
        ],
    )

    dataset = _use_case().execute(tmp_path)

    assert len(dataset.samples) == 1
    assert dataset.skipped == []
    row = dataset.samples[0].to_row()
    assert row["general.object_count"] == 2


def test_corrupt_pdf_is_skipped_not_fatal(tmp_path: Path) -> None:
    good_path = tmp_path / "doc-1.pdf"
    good_path.write_bytes(_valid_pdf_bytes())
    bad_path = tmp_path / "doc-2.pdf"
    bad_path.write_bytes(b"this is not a pdf at all")

    _write_benchmark_output(
        tmp_path,
        [_label_row("doc-1"), _label_row("doc-2")],
        [
            {
                "id": "doc-1",
                "path": str(good_path),
                "sha256": "a",
                "sha1": "b",
                "md5": "c",
                "blake2b": "d",
            },
            {
                "id": "doc-2",
                "path": str(bad_path),
                "sha256": "e",
                "sha1": "f",
                "md5": "g",
                "blake2b": "h",
            },
        ],
    )

    dataset = _use_case().execute(tmp_path)

    assert len(dataset.samples) == 1
    assert dataset.samples[0].id == "doc-1"
    assert len(dataset.skipped) == 1
    assert dataset.skipped[0].id == "doc-2"


def test_missing_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.pdf"
    _write_benchmark_output(
        tmp_path,
        [_label_row("doc-1")],
        [
            {
                "id": "doc-1",
                "path": str(missing_path),
                "sha256": "a",
                "sha1": "b",
                "md5": "c",
                "blake2b": "d",
            }
        ],
    )

    dataset = _use_case().execute(tmp_path)

    assert dataset.samples == []
    assert len(dataset.skipped) == 1
    assert dataset.skipped[0].id == "doc-1"
