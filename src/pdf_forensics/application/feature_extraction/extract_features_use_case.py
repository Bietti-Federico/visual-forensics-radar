"""Application boundary for turning a `PdfDocument` into a `FeatureSet`.

Takes its extractors as a constructor argument rather than importing a
default set from `plugins/` — the composition root (tests, a future CLI/API)
wires concrete extractors in, keeping this layer's only dependency on the
outer `plugins/` package pointing the correct direction: outward-in, never
inward-out (see `ports.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.feature_extraction.ports import FeatureExtractorPlugin
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.document import PdfDocument

#: Schema version stamped onto every `Feature` this use case produces.
#: Bump only when a feature's meaning changes incompatibly, not when a new
#: feature is merely added — existing feature stores key off this to decide
#: whether historical rows are still comparable to newly extracted ones.
SCHEMA_VERSION = "1.0.0"


class FeatureExtractionUseCase:
    def __init__(self, extractors: Sequence[FeatureExtractorPlugin]) -> None:
        self._extractors = tuple(extractors)

    def execute(self, document: PdfDocument) -> FeatureSet:
        features = []
        for extractor in self._extractors:
            features.extend(extractor.extract(document))
        return FeatureSet(features=features)
