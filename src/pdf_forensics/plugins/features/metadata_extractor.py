"""Document Information Dictionary features (ISO 32000-1 §14.3.3).

XMP metadata semantics are explicitly deferred (see `plugins/features/__init__.py`)
— this extractor only reads the classic `/Info` dictionary reachable from the
trailer.
"""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfHexString, PdfLiteralString
from pdf_forensics.plugins.features._shared import make_feature

_INFO_STRING_KEYS = ("Title", "Author", "Creator", "Producer", "CreationDate", "ModDate")


class MetadataFeatureExtractor:
    category = FeatureCategory.METADATA

    def extract(self, document: PdfDocument) -> list[Feature]:
        info = self._resolve_info_dict(document)

        if info is None:
            return [
                make_feature(
                    self.category,
                    "metadata.has_info_dict",
                    False,
                    FeatureType.BOOLEAN,
                    "Whether the trailer's /Info entry resolves to a dictionary.",
                    "PdfDocument.trailer./Info",
                ),
            ]

        features = [
            make_feature(
                self.category,
                "metadata.has_info_dict",
                True,
                FeatureType.BOOLEAN,
                "Whether the trailer's /Info entry resolves to a dictionary.",
                "PdfDocument.trailer./Info",
            ),
            make_feature(
                self.category,
                "metadata.key_count",
                len(info.entries),
                FeatureType.INTEGER,
                "Number of keys present in the /Info dictionary.",
                "len(Info.entries)",
            ),
        ]
        for key in _INFO_STRING_KEYS:
            value = self._string_value(info, key)
            snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in key).lstrip("_")
            features.append(
                make_feature(
                    self.category,
                    f"metadata.has_{snake}",
                    value is not None,
                    FeatureType.BOOLEAN,
                    f"Whether /Info has a /{key} entry decodable as text.",
                    f"Info./{key}",
                )
            )
            features.append(
                make_feature(
                    self.category,
                    f"metadata.{snake}",
                    value,
                    FeatureType.STRING,
                    f"The /Info /{key} value, decoded to text, or None if absent/undecodable.",
                    f"Info./{key}",
                )
            )
        return features

    def _resolve_info_dict(self, document: PdfDocument) -> PdfDictionary | None:
        info_ref = document.trailer.get_ref("Info")
        if info_ref is None:
            return None
        info_obj = document.resolve(info_ref)
        if info_obj is None or not isinstance(info_obj.value, PdfDictionary):
            return None
        return info_obj.value

    def _string_value(self, dictionary: PdfDictionary, key: str) -> str | None:
        value = dictionary.get(key)
        if isinstance(value, PdfLiteralString | PdfHexString):
            return value.decode_text()
        return None
