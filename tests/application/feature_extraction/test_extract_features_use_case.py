from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    SCHEMA_VERSION,
    FeatureExtractionUseCase,
)
from pdf_forensics.domain.features.enums import FeatureCategory
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder


def _rich_document():
    builder = PdfBuilder()
    off_info = builder.add_object(4, 0, "<< /Title (Report) /Producer (Acme) >>")
    off_pages = builder.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off_page = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R >>")
    off_catalog = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off_stream = builder.add_stream_object(5, 0, "<< /Length 5 >>", b"HELLO")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off_catalog, 0, "n"),
            (2, off_pages, 0, "n"),
            (3, off_page, 0, "n"),
            (4, off_info, 0, "n"),
            (5, off_stream, 0, "n"),
        ],
        size=6,
        root_ref="1 0 R",
        extra_trailer="/Info 4 0 R",
    )
    return PdfDocumentParser().parse(builder.build())


def test_execute_assembles_features_from_every_extractor() -> None:
    use_case = FeatureExtractionUseCase(default_feature_extractors())
    document = _rich_document()

    feature_set = use_case.execute(document)

    flat = feature_set.to_dict()
    assert flat["general.object_count"] == 5
    assert flat["metadata.title"] == "Report"
    assert flat["catalog.page_count"] == 1
    assert flat["streams.stream_count"] == 1
    assert flat["security.is_encrypted"] is False

    categories = {feature.category for feature in feature_set}
    assert categories == set(FeatureCategory)

    for feature in feature_set:
        assert feature.schema_version == SCHEMA_VERSION


def test_execute_produces_no_duplicate_feature_names() -> None:
    use_case = FeatureExtractionUseCase(default_feature_extractors())
    document = _rich_document()

    feature_set = use_case.execute(document)

    names = [feature.name for feature in feature_set]
    assert len(names) == len(set(names))
    # to_dict() would raise on a collision; confirm it doesn't.
    feature_set.to_dict()
