"""Incremental-update features: how many times the file was saved, per the /Prev chain."""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.plugins.features._shared import make_feature


class IncrementalUpdatesFeatureExtractor:
    category = FeatureCategory.INCREMENTAL_UPDATES

    def extract(self, document: PdfDocument) -> list[Feature]:
        revision_count = len(document.revisions)
        return [
            make_feature(
                self.category,
                "incremental_updates.revision_count",
                revision_count,
                FeatureType.INTEGER,
                "Number of revisions found by following /Prev from startxref.",
                "len(PdfDocument.revisions)",
            ),
            make_feature(
                self.category,
                "incremental_updates.has_incremental_updates",
                revision_count > 1,
                FeatureType.BOOLEAN,
                "Whether the file has been saved more than once (more than one revision).",
                "len(PdfDocument.revisions) > 1",
            ),
        ]
