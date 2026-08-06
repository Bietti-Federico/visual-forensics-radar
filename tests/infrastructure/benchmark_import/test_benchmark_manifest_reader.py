import csv
from pathlib import Path

import pytest

from pdf_forensics.infrastructure.benchmark_import.benchmark_manifest_reader import (
    read_benchmark_samples,
)

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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_benchmark_output(tmp_path: Path, label_rows: list[dict], hash_rows: list[dict]) -> Path:
    _write_csv(tmp_path / "labels.csv", _LABELS_FIELDS, label_rows)
    _write_csv(tmp_path / "hashes.csv", _HASHES_FIELDS, hash_rows)
    return tmp_path


def test_reads_a_seeded_root_row(tmp_path: Path) -> None:
    label_row = {
        "id": "doc-1",
        "seed": "42",
        "original_id": "doc-1",
        "parent_id": "",
        "generator": "receipt_generator",
        "pipeline": "",
        "transformation_count": "0",
        "pipeline_depth": "0",
        "is_original": "True",
        "binary_class": "original",
        "multiclass_label": "original",
    }
    hash_row = {
        "id": "doc-1",
        "path": "/data/doc-1.pdf",
        "sha256": "a",
        "sha1": "b",
        "md5": "c",
        "blake2b": "d",
    }
    directory = _write_benchmark_output(tmp_path, [label_row], [hash_row])

    refs = read_benchmark_samples(directory)

    assert len(refs) == 1
    ref = refs[0]
    assert ref.id == "doc-1"
    assert ref.path == Path("/data/doc-1.pdf")
    assert ref.seed == 42
    assert ref.parent_id is None
    assert ref.pipeline == ()
    assert ref.is_original is True
    assert ref.pipeline_depth == 0


def test_reads_an_external_pdf_row_with_no_seed(tmp_path: Path) -> None:
    label_row = {
        "id": "ext-1",
        "seed": "",
        "original_id": "ext-1",
        "parent_id": "",
        "generator": "external:my_template",
        "pipeline": "",
        "transformation_count": "0",
        "pipeline_depth": "0",
        "is_original": "True",
        "binary_class": "original",
        "multiclass_label": "external:my_template",
    }
    hash_row = {
        "id": "ext-1",
        "path": "/data/ext-1.pdf",
        "sha256": "a",
        "sha1": "b",
        "md5": "c",
        "blake2b": "d",
    }
    directory = _write_benchmark_output(tmp_path, [label_row], [hash_row])

    refs = read_benchmark_samples(directory)

    assert refs[0].seed is None
    assert refs[0].multiclass_label == "external:my_template"


def test_reads_a_transformed_child_row_with_pipeline(tmp_path: Path) -> None:
    label_row = {
        "id": "doc-1-child",
        "seed": "42",
        "original_id": "doc-1",
        "parent_id": "doc-1",
        "generator": "receipt_generator",
        "pipeline": "pikepdf_metadata_removal",
        "transformation_count": "1",
        "pipeline_depth": "1",
        "is_original": "False",
        "binary_class": "transformed",
        "multiclass_label": "pikepdf_metadata_removal",
    }
    hash_row = {
        "id": "doc-1-child",
        "path": "/data/doc-1-child.pdf",
        "sha256": "a",
        "sha1": "b",
        "md5": "c",
        "blake2b": "d",
    }
    directory = _write_benchmark_output(tmp_path, [label_row], [hash_row])

    refs = read_benchmark_samples(directory)

    assert refs[0].pipeline == ("pikepdf_metadata_removal",)
    assert refs[0].parent_id == "doc-1"
    assert refs[0].is_original is False


def test_missing_labels_csv_raises(tmp_path: Path) -> None:
    _write_csv(tmp_path / "hashes.csv", _HASHES_FIELDS, [])
    with pytest.raises(FileNotFoundError, match="labels.csv"):
        read_benchmark_samples(tmp_path)


def test_missing_hashes_csv_raises(tmp_path: Path) -> None:
    _write_csv(tmp_path / "labels.csv", _LABELS_FIELDS, [])
    with pytest.raises(FileNotFoundError, match="hashes.csv"):
        read_benchmark_samples(tmp_path)


def test_label_row_with_no_matching_hash_raises(tmp_path: Path) -> None:
    label_row = {
        "id": "orphan",
        "seed": "1",
        "original_id": "orphan",
        "parent_id": "",
        "generator": "receipt_generator",
        "pipeline": "",
        "transformation_count": "0",
        "pipeline_depth": "0",
        "is_original": "True",
        "binary_class": "original",
        "multiclass_label": "original",
    }
    directory = _write_benchmark_output(tmp_path, [label_row], [])

    with pytest.raises(ValueError, match="orphan"):
        read_benchmark_samples(directory)
