from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.catalog_extractor import CatalogFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_catalog_absent_when_root_unresolvable() -> None:
    builder = PdfBuilder()
    # /Root points at an object number that was never defined.
    builder.add_classic_xref_and_trailer([(0, 0, 65535, "f")], size=1, root_ref="9 0 R")
    document = PdfDocumentParser().parse(builder.build())

    features = CatalogFeatureExtractor().extract(document)

    assert get_feature(features, "catalog.has_catalog").value is False
    assert len(features) == 1


def test_catalog_present_with_pages_count_and_acroform() -> None:
    builder = PdfBuilder()
    off_pages = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 3 >>")
    off1 = builder.add_object(
        1, 0, "<< /Type /Catalog /Pages 2 0 R /Version /1.6 /AcroForm << >> >>"
    )
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off_pages, 0, "n")],
        size=3,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = CatalogFeatureExtractor().extract(document)

    assert get_feature(features, "catalog.has_catalog").value is True
    assert get_feature(features, "catalog.type_is_catalog").value is True
    assert get_feature(features, "catalog.version").value == "1.6"
    assert get_feature(features, "catalog.page_count").value == 3
    assert get_feature(features, "catalog.has_acroform").value is True
    assert get_feature(features, "catalog.has_outlines").value is False


def test_catalog_page_count_none_when_pages_unresolvable() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 9 0 R >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = CatalogFeatureExtractor().extract(document)

    assert get_feature(features, "catalog.page_count").value is None


def test_catalog_page_count_none_when_pages_key_absent() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = CatalogFeatureExtractor().extract(document)

    assert get_feature(features, "catalog.page_count").value is None


def test_catalog_absent_when_trailer_has_no_root_key() -> None:
    builder = PdfBuilder()
    builder.add_object(1, 0, "<< /Type /Catalog >>")
    # A trailer with no /Root at all (not just an unresolvable one).
    builder.add_raw(b"trailer\n<< /Size 2 >>\n")

    document = PdfDocumentParser().parse(builder.build())

    features = CatalogFeatureExtractor().extract(document)

    assert get_feature(features, "catalog.has_catalog").value is False
