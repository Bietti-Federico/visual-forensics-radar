# Training Data Ingestion (benchmark integration)

Covers `src/pdf_forensics/domain/training_data/`,
`src/pdf_forensics/infrastructure/benchmark_import/`,
`src/pdf_forensics/application/training_data/`, and `scripts/build_training_dataset.py`.

## Why this module exists

The sibling project `pdf-forensics-benchmark` ("el builder") generates synthetic
(and, by explicit developer decision during this development phase, real)
PDFs plus transformation variants, provenance, and ground-truth labels — meant
to train this platform's future ML modules (Anomaly Detection / ML Ensemble).

The benchmark's own `FeatureExtractor` (pikepdf/PyMuPDF/qpdf/exiftool-based)
computes a **different feature space** than this platform's own Module 2
(`FeatureExtractionUseCase`, built on this platform's from-scratch parser).
Training on the benchmark's features and running inference with this
platform's features would be a feature-space mismatch the model could never
bridge. This module closes that gap: it re-parses and re-featurizes every
benchmark-generated PDF with this platform's *own* pipeline (Modules 1-2),
joined with the benchmark's ground-truth labels.

**Scope**: this only *prepares* the dataset (features + label). It does not
train anything — that's a future ML Ensemble module.

## The two projects stay decoupled

This module reads only two plain CSV files pdf-forensics-benchmark's
`DatasetBuilder.build()` writes — `labels.csv` (ground truth) and
`hashes.csv` (where each generated PDF lives on disk), joined on `id`. There
is no dependency on that project's Python package, only on its output file
format (`infrastructure/benchmark_import/benchmark_manifest_reader.py` is the
one place that knows those column names).

## Usage

```bash
# 1. Generate a benchmark run (in pdf-forensics-benchmark):
poetry run python examples/build_dataset_with_external_pdfs.py my_document.pdf

# 2. Re-featurize it in this platform's own feature space:
poetry run python scripts/build_training_dataset.py \
    ../pdf-forensics-benchmark/examples/output/dataset_with_external_pdfs \
    training_dataset.csv
```

The resulting CSV has one row per benchmark-generated document (original +
every transformed variant), with columns from this platform's own
`FeatureSet` (`general.*`, `metadata.*`, `xref.*`, ...) plus the label
columns (`is_original`, `binary_class`, `multiclass_label`, `pipeline_depth`,
`generator`, `pipeline`, `seed`, `original_id`, `parent_id`).

## Tolerant of individual bad samples

A benchmark transformation could in principle produce a PDF this platform's
parser can't handle. `BuildTrainingDatasetUseCase` catches `NotAPdfError`,
`UnrecoverableStructureError`, and `OSError` (moved/deleted file) per sample,
recording a `SkippedSample` (id, path, reason) rather than aborting the whole
batch — the same tolerant philosophy Module 1 uses for structural anomalies.

A mismatch between `labels.csv` and `hashes.csv` (an `id` in one but not the
other) is treated differently: that means the two files don't actually come
from the same run, which is a setup error worth surfacing immediately, so
`read_benchmark_samples` raises rather than silently skipping.
