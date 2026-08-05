"""Trailer-dictionary features (ISO 32000-1 §7.5.5) — the merged, latest-wins trailer."""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.plugins.features._shared import make_feature


class TrailerFeatureExtractor:
    category = FeatureCategory.TRAILER

    def extract(self, document: PdfDocument) -> list[Feature]:
        trailer = document.trailer
        return [
            make_feature(
                self.category,
                "trailer.key_count",
                len(trailer.entries),
                FeatureType.INTEGER,
                "Number of keys in the merged (most-recent-revision-wins) trailer.",
                "len(PdfDocument.trailer.entries)",
            ),
            make_feature(
                self.category,
                "trailer.has_root",
                trailer.get_ref("Root") is not None,
                FeatureType.BOOLEAN,
                "Whether the trailer has a /Root entry.",
                "trailer./Root",
            ),
            make_feature(
                self.category,
                "trailer.has_info",
                trailer.get_ref("Info") is not None,
                FeatureType.BOOLEAN,
                "Whether the trailer has an /Info entry.",
                "trailer./Info",
            ),
            make_feature(
                self.category,
                "trailer.has_id",
                trailer.get("ID") is not None,
                FeatureType.BOOLEAN,
                "Whether the trailer has an /ID entry.",
                "trailer./ID",
            ),
            make_feature(
                self.category,
                "trailer.has_encrypt",
                trailer.get("Encrypt") is not None,
                FeatureType.BOOLEAN,
                "Whether the trailer has an /Encrypt entry.",
                "trailer./Encrypt",
            ),
        ]
