from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.xref_extractor import XrefFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_classic_table_counts() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = XrefFeatureExtractor().extract(document)

    assert get_feature(features, "xref.uses_xref_stream").value is False
    assert get_feature(features, "xref.uses_hybrid_xref").value is False
    assert get_feature(features, "xref.free_entry_count").value == 1
    assert get_feature(features, "xref.in_use_entry_count").value == 1
    assert get_feature(features, "xref.compressed_entry_count").value == 0


def test_xref_stream_and_compressed_entries() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    objstm_offset = builder.add_object_stream(
        container_obj_num=3, packed_objects=[(4, "<< /Type /Page >>")]
    )
    builder.add_xref_stream(
        obj_num=2,
        entries={1: (1, off1, 0), 3: (1, objstm_offset, 0), 4: (2, 3, 0)},
        size=5,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = XrefFeatureExtractor().extract(document)

    assert get_feature(features, "xref.uses_xref_stream").value is True
    assert get_feature(features, "xref.compressed_entry_count").value == 1
