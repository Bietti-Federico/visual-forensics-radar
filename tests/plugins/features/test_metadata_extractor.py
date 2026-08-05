from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.metadata_extractor import MetadataFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_metadata_absent_when_no_info_dict() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = MetadataFeatureExtractor().extract(document)

    assert get_feature(features, "metadata.has_info_dict").value is False
    assert len(features) == 1


def test_metadata_absent_when_info_ref_unresolvable() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")],
        size=2,
        root_ref="1 0 R",
        extra_trailer="/Info 9 0 R",  # object 9 was never defined
    )
    document = PdfDocumentParser().parse(builder.build())

    features = MetadataFeatureExtractor().extract(document)

    assert get_feature(features, "metadata.has_info_dict").value is False


def test_metadata_present_decodes_info_dict_strings() -> None:
    builder = PdfBuilder()
    off_info = builder.add_object(2, 0, "<< /Title (Test Doc) /Author (Jane) /Producer (Acme) >>")
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off_info, 0, "n")],
        size=3,
        root_ref="1 0 R",
        extra_trailer="/Info 2 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = MetadataFeatureExtractor().extract(document)

    assert get_feature(features, "metadata.has_info_dict").value is True
    assert get_feature(features, "metadata.key_count").value == 3
    assert get_feature(features, "metadata.title").value == "Test Doc"
    assert get_feature(features, "metadata.has_title").value is True
    assert get_feature(features, "metadata.author").value == "Jane"
    assert get_feature(features, "metadata.has_creation_date").value is False
    assert get_feature(features, "metadata.creation_date").value is None
