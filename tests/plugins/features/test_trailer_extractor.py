from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.trailer_extractor import TrailerFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_trailer_features_minimal() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = TrailerFeatureExtractor().extract(document)

    assert get_feature(features, "trailer.has_root").value is True
    assert get_feature(features, "trailer.has_info").value is False
    assert get_feature(features, "trailer.has_id").value is False
    assert get_feature(features, "trailer.has_encrypt").value is False
    assert get_feature(features, "trailer.key_count").value == 2  # /Size /Root


def test_trailer_features_with_id_and_info() -> None:
    builder = PdfBuilder()
    off_info = builder.add_object(2, 0, "<< /Title (t) >>")
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off_info, 0, "n")],
        size=3,
        root_ref="1 0 R",
        extra_trailer="/Info 2 0 R /ID [<AA> <BB>]",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = TrailerFeatureExtractor().extract(document)

    assert get_feature(features, "trailer.has_info").value is True
    assert get_feature(features, "trailer.has_id").value is True
