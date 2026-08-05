from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.statistics_extractor import StatisticsFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_no_anomalies() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = StatisticsFeatureExtractor().extract(document)

    assert get_feature(features, "statistics.anomaly_count_total").value == 0
    assert get_feature(features, "statistics.anomaly_count_warning").value == 0
    assert get_feature(features, "statistics.anomaly_code_histogram").value == {}


def test_duplicate_object_id_anomaly_is_counted() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    off1_dup = builder.add_object(1, 0, "<< /Type /Catalog /Extra true >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (1, off1_dup, 0, "n")],
        size=2,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = StatisticsFeatureExtractor().extract(document)

    assert get_feature(features, "statistics.anomaly_count_total").value == 1
    assert get_feature(features, "statistics.anomaly_count_warning").value == 1
    assert get_feature(features, "statistics.anomaly_code_histogram").value == {
        "duplicate_object_id": 1
    }
