from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.objects_extractor import ObjectsFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_type_histogram_and_no_duplicates() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "[1 2 3]")
    off3 = builder.add_object(3, 0, "/Foo")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = ObjectsFeatureExtractor().extract(document)

    histogram = get_feature(features, "objects.type_histogram").value
    assert histogram["dictionary"] == 1
    assert histogram["array"] == 1
    assert histogram["name"] == 1
    assert get_feature(features, "objects.duplicate_object_id_count").value == 0


def test_duplicate_object_id_count_reflects_anomalies() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    off1_dup = builder.add_object(1, 0, "<< /Type /Catalog /Extra true >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (1, off1_dup, 0, "n")],
        size=2,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = ObjectsFeatureExtractor().extract(document)

    assert get_feature(features, "objects.duplicate_object_id_count").value == 1
