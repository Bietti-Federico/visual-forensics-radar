"""Reads a `pdf-forensics-benchmark` `DatasetBuilder.build()` output directory.

Only two of that run's output files are needed — `labels.csv` (ground truth)
and `hashes.csv` (where each generated PDF actually lives on disk) — joined
on `id`. The benchmark's own `features.csv` (a different feature space, see
`application/training_data/build_training_dataset_use_case.py`'s module
docstring) is intentionally never read here.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pdf_forensics.domain.training_data.benchmark_sample_ref import BenchmarkSampleRef

_LABELS_FILENAME = "labels.csv"
_HASHES_FILENAME = "hashes.csv"


def read_benchmark_samples(benchmark_output_dir: Path) -> list[BenchmarkSampleRef]:
    labels_path = benchmark_output_dir / _LABELS_FILENAME
    hashes_path = benchmark_output_dir / _HASHES_FILENAME
    if not labels_path.is_file():
        raise FileNotFoundError(f"No {_LABELS_FILENAME} found in {benchmark_output_dir}")
    if not hashes_path.is_file():
        raise FileNotFoundError(f"No {_HASHES_FILENAME} found in {benchmark_output_dir}")

    paths_by_id = _read_paths_by_id(hashes_path)

    refs = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample_id = row["id"]
            if sample_id not in paths_by_id:
                raise ValueError(
                    f"labels.csv references id {sample_id!r} with no matching row in "
                    f"{_HASHES_FILENAME} — {benchmark_output_dir} looks inconsistent."
                )
            refs.append(_build_ref(row, paths_by_id[sample_id]))
    return refs


def _read_paths_by_id(hashes_path: Path) -> dict[str, Path]:
    with hashes_path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: Path(row["path"]) for row in csv.DictReader(f)}


def _build_ref(row: dict[str, str], path: Path) -> BenchmarkSampleRef:
    return BenchmarkSampleRef(
        id=row["id"],
        path=path,
        seed=int(row["seed"]) if row["seed"] else None,
        original_id=row["original_id"],
        parent_id=row["parent_id"] or None,
        generator=row["generator"],
        pipeline=tuple(step for step in row["pipeline"].split("|") if step),
        pipeline_depth=int(row["pipeline_depth"]),
        is_original=row["is_original"] == "True",
        binary_class=row["binary_class"],
        multiclass_label=row["multiclass_label"],
    )
