"""One file from the production training corpus (see
`infrastructure/training_corpus/`), re-parsed/re-featurized with this
platform's own pipeline — same reasoning as
`domain/training_data/training_sample.py`, but for a plain filesystem
corpus (`training_corpus/genuine|confirmed_fraud/<entity>/*.pdf`) rather
than a `pdf-forensics-benchmark` output directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pdf_forensics.domain.features.feature_set import FeatureSet


@dataclass(frozen=True, slots=True)
class TrainingCorpusEntry:
    entity: str
    is_genuine: bool
    path: Path
    features: FeatureSet


@dataclass(frozen=True, slots=True)
class SkippedCorpusEntry:
    entity: str
    path: Path
    reason: str
