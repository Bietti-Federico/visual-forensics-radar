"""Build a training CSV, in the validator's own feature space, from a benchmark run.

Usage:
    poetry run python scripts/build_training_dataset.py <benchmark_output_dir> <output_csv>

`<benchmark_output_dir>` is a directory previously produced by
pdf-forensics-benchmark's `DatasetBuilder.build()` (e.g. via its
`examples/build_dataset_with_external_pdfs.py`). This script re-parses and
re-featurizes every PDF it references with this platform's own pipeline and
writes the result — features plus ground-truth labels — to `<output_csv>`.

This is a practical entry point, not the future Feature Store module
(a versioned, multi-backend — CSV/Parquet/SQLite/DuckDB — persistence layer);
deliberately minimal.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.training_data.build_training_dataset_use_case import (
    BuildTrainingDatasetUseCase,
)
from pdf_forensics.plugins.features import default_feature_extractors


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    benchmark_output_dir = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])

    use_case = BuildTrainingDatasetUseCase(
        parse_pdf_use_case=ParsePdfUseCase(),
        feature_extraction_use_case=FeatureExtractionUseCase(default_feature_extractors()),
    )
    dataset = use_case.execute(benchmark_output_dir)

    _write_csv(dataset.to_rows(), output_csv)

    print(f"Samples written: {len(dataset.samples)}")
    print(f"Skipped: {len(dataset.skipped)}")
    for skipped in dataset.skipped:
        print(f"  {skipped.id} ({skipped.path}): {skipped.reason}")
    print(f"Output: {output_csv}")


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: dict[str, None] = {}
    for row in rows:
        for key in row:
            fieldnames.setdefault(key, None)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
