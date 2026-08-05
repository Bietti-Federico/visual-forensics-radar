"""General file-level features: version, size, and gross object/revision counts."""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.plugins.features._shared import make_feature


class GeneralFeatureExtractor:
    category = FeatureCategory.GENERAL

    def extract(self, document: PdfDocument) -> list[Feature]:
        return [
            make_feature(
                self.category,
                "general.pdf_version",
                document.pdf_version,
                FeatureType.STRING,
                "The version declared in the file's %PDF-x.y header, or None if unparseable.",
                "PdfDocument.pdf_version",
            ),
            make_feature(
                self.category,
                "general.file_size_bytes",
                document.raw_size,
                FeatureType.INTEGER,
                "Total size of the file in bytes.",
                "PdfDocument.raw_size",
            ),
            make_feature(
                self.category,
                "general.object_count",
                len(document.objects),
                FeatureType.INTEGER,
                "Number of distinct indirect objects resolved across all revisions.",
                "len(PdfDocument.objects)",
            ),
            make_feature(
                self.category,
                "general.revision_count",
                len(document.revisions),
                FeatureType.INTEGER,
                "Number of xref/trailer revisions found by following /Prev.",
                "len(PdfDocument.revisions)",
            ),
        ]
