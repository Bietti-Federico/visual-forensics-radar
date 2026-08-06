"""A typed reference to one document a benchmark run generated, plus its ground truth.

The parsed, typed form of one joined `labels.csv` + `hashes.csv` row from a
`pdf-forensics-benchmark` output directory — see
`infrastructure/benchmark_import/benchmark_manifest_reader.py` for how it's
built. Deliberately holds no dependency on that project's own Python code,
only on the shape of its CSV output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkSampleRef:
    id: str
    path: Path
    seed: int | None
    original_id: str
    parent_id: str | None
    generator: str
    pipeline: tuple[str, ...]
    pipeline_depth: int
    is_original: bool
    binary_class: str
    multiclass_label: str
