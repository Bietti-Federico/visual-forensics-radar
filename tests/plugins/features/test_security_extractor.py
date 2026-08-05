from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.security_extractor import SecurityFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_not_encrypted() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = SecurityFeatureExtractor().extract(document)

    assert get_feature(features, "security.is_encrypted").value is False
    assert len(features) == 1


def test_encrypted_reads_raw_encryption_dict_fields() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    off_encrypt = builder.add_object(2, 0, "<< /Filter /Standard /V 2 /R 3 >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off_encrypt, 0, "n")],
        size=3,
        root_ref="1 0 R",
        extra_trailer="/Encrypt 2 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = SecurityFeatureExtractor().extract(document)

    assert get_feature(features, "security.is_encrypted").value is True
    assert get_feature(features, "security.encryption_filter_name").value == "Standard"
    assert get_feature(features, "security.encryption_v").value == 2
    assert get_feature(features, "security.encryption_r").value == 3
