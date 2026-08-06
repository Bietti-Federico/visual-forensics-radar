"""Builds an ML training table from a `pdf-forensics-benchmark` output directory.

The benchmark's own `FeatureExtractor` (pikepdf/PyMuPDF/qpdf/exiftool-based)
computes a *different* feature space than this platform's own Module 2
(`FeatureExtractionUseCase`, built on this platform's from-scratch parser).
Training on the benchmark's features and then running inference with this
platform's features would be a feature-space mismatch the model could never
bridge. This use case closes that gap: it re-parses and re-featurizes every
benchmark-generated PDF with this platform's *own* pipeline (Modules 1-2),
joined with the benchmark's ground-truth labels — producing a training table
in the exact feature space this platform uses at inference time.

Scope: this only *prepares* the dataset. It does not train anything — that's
a future ML Ensemble module.
"""

from __future__ import annotations

from pathlib import Path

from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.domain.pdf.errors import NotAPdfError, UnrecoverableStructureError
from pdf_forensics.domain.training_data.training_sample import (
    SkippedSample,
    TrainingDataset,
    TrainingSample,
)
from pdf_forensics.infrastructure.benchmark_import.benchmark_manifest_reader import (
    read_benchmark_samples,
)


class BuildTrainingDatasetUseCase:
    def __init__(
        self,
        parse_pdf_use_case: ParsePdfUseCase,
        feature_extraction_use_case: FeatureExtractionUseCase,
    ) -> None:
        self._parse_pdf = parse_pdf_use_case
        self._extract_features = feature_extraction_use_case

    def execute(self, benchmark_output_dir: Path) -> TrainingDataset:
        dataset = TrainingDataset()
        for ref in read_benchmark_samples(benchmark_output_dir):
            try:
                data = ref.path.read_bytes()
                document = self._parse_pdf.execute(data)
                feature_set = self._extract_features.execute(document)
            except (NotAPdfError, UnrecoverableStructureError, OSError) as exc:
                dataset.skipped.append(SkippedSample(id=ref.id, path=ref.path, reason=str(exc)))
                continue
            dataset.samples.append(TrainingSample(id=ref.id, features=feature_set, source=ref))
        return dataset
