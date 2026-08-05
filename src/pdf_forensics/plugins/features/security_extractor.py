"""Encryption-flag features. No decryption is performed — see `PdfDocument.encryption_dict`."""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.plugins.features._shared import make_feature


class SecurityFeatureExtractor:
    category = FeatureCategory.SECURITY

    def extract(self, document: PdfDocument) -> list[Feature]:
        features = [
            make_feature(
                self.category,
                "security.is_encrypted",
                document.is_encrypted,
                FeatureType.BOOLEAN,
                "Whether the trailer has an /Encrypt entry. No decryption is attempted.",
                "PdfDocument.is_encrypted",
            ),
        ]

        encryption_dict = document.encryption_dict
        if encryption_dict is not None:
            features.append(
                make_feature(
                    self.category,
                    "security.encryption_filter_name",
                    encryption_dict.get_name("Filter"),
                    FeatureType.STRING,
                    "The encryption handler's /Filter name (e.g. /Standard), read raw "
                    "from the undecrypted /Encrypt dictionary.",
                    "PdfDocument.encryption_dict./Filter",
                )
            )
            features.append(
                make_feature(
                    self.category,
                    "security.encryption_v",
                    encryption_dict.get_int("V"),
                    FeatureType.INTEGER,
                    "The encryption dictionary's /V (algorithm version) entry.",
                    "PdfDocument.encryption_dict./V",
                )
            )
            features.append(
                make_feature(
                    self.category,
                    "security.encryption_r",
                    encryption_dict.get_int("R"),
                    FeatureType.INTEGER,
                    "The encryption dictionary's /R (revision) entry.",
                    "PdfDocument.encryption_dict./R",
                )
            )
        return features
