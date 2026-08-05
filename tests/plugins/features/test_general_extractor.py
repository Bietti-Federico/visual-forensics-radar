from pdf_forensics.domain.features.enums import FeatureCategory
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.general_extractor import GeneralFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def _minimal_document():
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")],
        size=3,
        root_ref="1 0 R",
    )
    return PdfDocumentParser().parse(builder.build())


def test_general_features() -> None:
    document = _minimal_document()
    features = GeneralFeatureExtractor().extract(document)

    assert get_feature(features, "general.pdf_version").value == "1.7"
    assert get_feature(features, "general.file_size_bytes").value == document.raw_size
    assert get_feature(features, "general.object_count").value == 2
    assert get_feature(features, "general.revision_count").value == 1


def test_category_attribute() -> None:
    assert GeneralFeatureExtractor().category is FeatureCategory.GENERAL
