from pdf_forensics.domain.pdf.anomalies import (
    AnomalyCode,
    AnomalyCollector,
    AnomalySeverity,
)
from pdf_forensics.domain.pdf.objects import PdfReference


def test_collector_records_and_lists_anomalies() -> None:
    collector = AnomalyCollector()
    collector.record(
        AnomalyCode.DUPLICATE_OBJECT_ID,
        AnomalySeverity.WARNING,
        "duplicate",
        object_ref=PdfReference(3, 0),
        byte_offset=42,
    )

    anomalies = collector.to_list()
    assert len(anomalies) == 1
    anomaly = anomalies[0]
    assert anomaly.code == AnomalyCode.DUPLICATE_OBJECT_ID
    assert anomaly.severity == AnomalySeverity.WARNING
    assert anomaly.message == "duplicate"
    assert anomaly.object_ref == PdfReference(3, 0)
    assert anomaly.byte_offset == 42


def test_to_list_returns_a_copy() -> None:
    collector = AnomalyCollector()
    collector.record(AnomalyCode.TRAILER_MISSING_ROOT, AnomalySeverity.INFO, "missing")
    snapshot = collector.to_list()
    collector.record(AnomalyCode.BROKEN_PREV_CHAIN, AnomalySeverity.INFO, "cycle")
    assert len(snapshot) == 1
    assert len(collector.to_list()) == 2
