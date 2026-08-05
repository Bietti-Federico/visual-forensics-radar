"""The port `FeatureExtractionUseCase` depends on; `plugins/features/` implements it.

`FeatureExtractorPlugin` is a `Protocol` (structural typing), not an ABC — an
extractor needs no import of this module to satisfy it at runtime, only to be
type-checked against it. Concrete extractors live in the outer `plugins/`
layer and depend inward on this port, never the other way around: this
module and the use case that depends on it know nothing about any specific
extractor.
"""

from __future__ import annotations

from typing import Protocol

from pdf_forensics.domain.features.enums import FeatureCategory
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument


class FeatureExtractorPlugin(Protocol):
    category: FeatureCategory

    def extract(self, document: PdfDocument) -> list[Feature]:
        """Return every feature this plugin can produce for `document`.

        Must never raise for a well-formed `PdfDocument` — a category with
        nothing to report (e.g. no `/Info` dictionary) returns features that
        say so (`has_info_dict: False`) rather than an empty list or an
        exception, so a caller can always tell "absent" from "not run."
        """
        ...
