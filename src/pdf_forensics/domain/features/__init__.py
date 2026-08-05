"""Domain model for forensic features extracted from an already-parsed `PdfDocument`.

Free of parsing/pikepdf/PyMuPDF-style dependencies, same as `domain/pdf/` —
these types describe *what a feature is*, never how it was computed.
"""

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature, FeatureValue
from pdf_forensics.domain.features.feature_set import FeatureSet

__all__ = [
    "Feature",
    "FeatureCategory",
    "FeatureSet",
    "FeatureType",
    "FeatureValue",
]
