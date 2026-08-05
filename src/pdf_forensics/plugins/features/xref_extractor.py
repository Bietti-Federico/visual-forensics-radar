"""Cross-reference features from the most recent revision's xref section."""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument, XrefEntryType
from pdf_forensics.plugins.features._shared import make_feature


class XrefFeatureExtractor:
    category = FeatureCategory.XREF

    def extract(self, document: PdfDocument) -> list[Feature]:
        latest = document.latest_revision
        entries = latest.xref_entries.values()
        counts = {entry_type: 0 for entry_type in XrefEntryType}
        for entry in entries:
            counts[entry.entry_type] += 1

        return [
            make_feature(
                self.category,
                "xref.uses_xref_stream",
                latest.is_xref_stream,
                FeatureType.BOOLEAN,
                "Whether the most recent revision uses a cross-reference stream (PDF 1.5+) "
                "rather than a classic xref table.",
                "PdfDocument.latest_revision.is_xref_stream",
            ),
            make_feature(
                self.category,
                "xref.uses_hybrid_xref",
                latest.trailer.get_int("XRefStm") is not None,
                FeatureType.BOOLEAN,
                "Whether the most recent revision's trailer has a /XRefStm entry "
                "(a hybrid-reference file, ISO 32000-1 §7.5.8.4).",
                "PdfDocument.latest_revision.trailer./XRefStm",
            ),
            make_feature(
                self.category,
                "xref.free_entry_count",
                counts[XrefEntryType.FREE],
                FeatureType.INTEGER,
                "Number of free (unused) entries in the most recent revision's xref section.",
                "PdfDocument.latest_revision.xref_entries",
            ),
            make_feature(
                self.category,
                "xref.in_use_entry_count",
                counts[XrefEntryType.IN_USE],
                FeatureType.INTEGER,
                "Number of in-use entries in the most recent revision's xref section.",
                "PdfDocument.latest_revision.xref_entries",
            ),
            make_feature(
                self.category,
                "xref.compressed_entry_count",
                counts[XrefEntryType.COMPRESSED],
                FeatureType.INTEGER,
                "Number of compressed-object entries in the most recent revision's "
                "xref section.",
                "PdfDocument.latest_revision.xref_entries",
            ),
        ]
