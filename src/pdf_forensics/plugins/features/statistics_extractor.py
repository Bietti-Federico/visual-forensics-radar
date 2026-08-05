"""Aggregate statistics over every structural anomaly the parser recorded."""

from __future__ import annotations

from collections import Counter

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.plugins.features._shared import make_feature


class StatisticsFeatureExtractor:
    category = FeatureCategory.STATISTICS

    def extract(self, document: PdfDocument) -> list[Feature]:
        anomalies = document.anomalies
        severity_counts = {severity: 0 for severity in AnomalySeverity}
        code_histogram: Counter[str] = Counter()
        for anomaly in anomalies:
            severity_counts[anomaly.severity] += 1
            code_histogram[anomaly.code.value] += 1

        return [
            make_feature(
                self.category,
                "statistics.anomaly_count_total",
                len(anomalies),
                FeatureType.INTEGER,
                "Total number of structural anomalies recorded during parsing.",
                "len(PdfDocument.anomalies)",
            ),
            make_feature(
                self.category,
                "statistics.anomaly_count_info",
                severity_counts[AnomalySeverity.INFO],
                FeatureType.INTEGER,
                "Number of INFO-severity anomalies.",
                "PdfDocument.anomalies",
            ),
            make_feature(
                self.category,
                "statistics.anomaly_count_warning",
                severity_counts[AnomalySeverity.WARNING],
                FeatureType.INTEGER,
                "Number of WARNING-severity anomalies.",
                "PdfDocument.anomalies",
            ),
            make_feature(
                self.category,
                "statistics.anomaly_count_critical",
                severity_counts[AnomalySeverity.CRITICAL],
                FeatureType.INTEGER,
                "Number of CRITICAL-severity anomalies.",
                "PdfDocument.anomalies",
            ),
            make_feature(
                self.category,
                "statistics.anomaly_code_histogram",
                dict(code_histogram),
                FeatureType.DICT,
                "Count of anomalies per AnomalyCode.",
                "PdfDocument.anomalies",
            ),
        ]
