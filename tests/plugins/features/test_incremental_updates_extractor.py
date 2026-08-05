from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.incremental_updates_extractor import (
    IncrementalUpdatesFeatureExtractor,
)
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_single_revision() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = IncrementalUpdatesFeatureExtractor().extract(document)

    assert get_feature(features, "incremental_updates.revision_count").value == 1
    assert get_feature(features, "incremental_updates.has_incremental_updates").value is False


def test_multiple_revisions() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    first_xref = builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    off1b = builder.add_object(1, 0, "<< /Type /Catalog /Extra true >>")
    builder.add_classic_xref_and_trailer(
        [(1, off1b, 0, "n")], size=2, root_ref="1 0 R", prev=first_xref
    )
    document = PdfDocumentParser().parse(builder.build())

    features = IncrementalUpdatesFeatureExtractor().extract(document)

    assert get_feature(features, "incremental_updates.revision_count").value == 2
    assert get_feature(features, "incremental_updates.has_incremental_updates").value is True
