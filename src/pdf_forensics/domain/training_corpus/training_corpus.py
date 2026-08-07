"""The complete, re-featurized production training corpus."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from pdf_forensics.domain.training_corpus.training_corpus_entry import (
    SkippedCorpusEntry,
    TrainingCorpusEntry,
)


@dataclass(slots=True)
class TrainingCorpus:
    entries: list[TrainingCorpusEntry] = field(default_factory=list)
    skipped: list[SkippedCorpusEntry] = field(default_factory=list)

    def by_entity(self) -> dict[str, list[TrainingCorpusEntry]]:
        grouped: dict[str, list[TrainingCorpusEntry]] = defaultdict(list)
        for entry in self.entries:
            grouped[entry.entity].append(entry)
        return dict(grouped)
