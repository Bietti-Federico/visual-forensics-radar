"""Parses + featurizes every file in the production training corpus.

Mirrors `application/training_data/build_training_dataset_use_case.py`'s
tolerance philosophy exactly (skip and record, never abort the whole batch
over one bad file) but reads the plain filesystem corpus
(`infrastructure/training_corpus/`) instead of a `pdf-forensics-benchmark`
output directory.
"""

from __future__ import annotations

from pathlib import Path

from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.domain.pdf.errors import NotAPdfError, UnrecoverableStructureError
from pdf_forensics.domain.training_corpus.training_corpus import TrainingCorpus
from pdf_forensics.domain.training_corpus.training_corpus_entry import (
    SkippedCorpusEntry,
    TrainingCorpusEntry,
)
from pdf_forensics.infrastructure.training_corpus.filesystem_training_corpus_reader import (
    read_training_corpus_files,
)


class BuildTrainingCorpusUseCase:
    def __init__(
        self,
        parse_pdf_use_case: ParsePdfUseCase,
        feature_extraction_use_case: FeatureExtractionUseCase,
    ) -> None:
        self._parse_pdf = parse_pdf_use_case
        self._extract_features = feature_extraction_use_case

    def execute(self, corpus_dir: Path) -> TrainingCorpus:
        corpus = TrainingCorpus()
        for ref in read_training_corpus_files(corpus_dir):
            try:
                data = ref.path.read_bytes()
                document = self._parse_pdf.execute(data)
                feature_set = self._extract_features.execute(document)
            except (NotAPdfError, UnrecoverableStructureError, OSError) as exc:
                corpus.skipped.append(
                    SkippedCorpusEntry(entity=ref.entity, path=ref.path, reason=str(exc))
                )
                continue
            corpus.entries.append(
                TrainingCorpusEntry(
                    entity=ref.entity,
                    is_genuine=ref.is_genuine,
                    path=ref.path,
                    features=feature_set,
                )
            )
        return corpus
