"""Scans the production training corpus directory tree:

    <corpus_dir>/
      genuine/<ENTITY>/*.pdf
      confirmed_fraud/<ENTITY>/*.pdf

The entity name is the folder name — deliberately no manifest/CSV, unlike
`infrastructure/benchmark_import/`. Adding a new entity is just a new
folder; adding a document is just a new file. Neither needs a code change
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_GENUINE_DIRNAME = "genuine"
_CONFIRMED_FRAUD_DIRNAME = "confirmed_fraud"
_PDF_GLOB = "*.pdf"


@dataclass(frozen=True, slots=True)
class TrainingFileRef:
    entity: str
    is_genuine: bool
    path: Path


def read_training_corpus_files(corpus_dir: Path) -> list[TrainingFileRef]:
    refs: list[TrainingFileRef] = []
    for is_genuine, dirname in ((True, _GENUINE_DIRNAME), (False, _CONFIRMED_FRAUD_DIRNAME)):
        root = corpus_dir / dirname
        if not root.is_dir():
            continue
        for entity_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for pdf_path in sorted(entity_dir.glob(_PDF_GLOB)):
                refs.append(
                    TrainingFileRef(entity=entity_dir.name, is_genuine=is_genuine, path=pdf_path)
                )
    return refs
