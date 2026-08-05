"""Object-graph features: a COS-type histogram and duplicate-id signal.

Only top-level object *values* are inspected (`isinstance` over
`PdfDocument.objects.values()`) — no recursion into arrays/dictionaries, same
scope boundary as the rest of this module.
"""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.anomalies import AnomalyCode
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.domain.pdf.objects import COS_TYPE_NAMES, cos_type_name
from pdf_forensics.plugins.features._shared import make_feature


class ObjectsFeatureExtractor:
    category = FeatureCategory.OBJECTS

    def extract(self, document: PdfDocument) -> list[Feature]:
        histogram = dict.fromkeys(COS_TYPE_NAMES, 0)
        for obj in document.objects.values():
            histogram[cos_type_name(obj.value)] += 1

        duplicate_count = sum(
            1 for anomaly in document.anomalies if anomaly.code is AnomalyCode.DUPLICATE_OBJECT_ID
        )

        return [
            make_feature(
                self.category,
                "objects.type_histogram",
                histogram,
                FeatureType.DICT,
                "Count of top-level resolved objects per COS type.",
                "PdfDocument.objects (isinstance histogram)",
            ),
            make_feature(
                self.category,
                "objects.duplicate_object_id_count",
                duplicate_count,
                FeatureType.INTEGER,
                "Number of DUPLICATE_OBJECT_ID structural anomalies detected.",
                "PdfDocument.anomalies",
            ),
        ]
