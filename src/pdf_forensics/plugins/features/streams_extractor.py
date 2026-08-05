"""Stream features: how many, which filters they claim, and total raw size."""

from __future__ import annotations

from collections import Counter

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.anomalies import AnomalyCode
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.domain.pdf.objects import PdfStream
from pdf_forensics.plugins.features._shared import make_feature


class StreamsFeatureExtractor:
    category = FeatureCategory.STREAMS

    def extract(self, document: PdfDocument) -> list[Feature]:
        streams = [
            obj.value for obj in document.objects.values() if isinstance(obj.value, PdfStream)
        ]
        filter_histogram: Counter[str] = Counter()
        for stream in streams:
            filter_histogram.update(stream.filter_names)

        unsupported_filter_count = sum(
            1 for anomaly in document.anomalies if anomaly.code is AnomalyCode.UNSUPPORTED_FILTER
        )

        return [
            make_feature(
                self.category,
                "streams.stream_count",
                len(streams),
                FeatureType.INTEGER,
                "Number of top-level resolved stream objects.",
                "PdfDocument.objects (isinstance PdfStream)",
            ),
            make_feature(
                self.category,
                "streams.filter_histogram",
                dict(filter_histogram),
                FeatureType.DICT,
                "Count of streams claiming each filter name (a stream with multiple "
                "filters is counted once per filter).",
                "PdfStream.filter_names",
            ),
            make_feature(
                self.category,
                "streams.unsupported_filter_anomaly_count",
                unsupported_filter_count,
                FeatureType.INTEGER,
                "Number of UNSUPPORTED_FILTER structural anomalies detected.",
                "PdfDocument.anomalies",
            ),
            make_feature(
                self.category,
                "streams.total_raw_bytes",
                sum(len(stream.raw_data) for stream in streams),
                FeatureType.INTEGER,
                "Total size, in bytes, of all stream raw_data (still filter-encoded "
                "where a filter wasn't decoded).",
                "sum(len(PdfStream.raw_data))",
            ),
        ]
