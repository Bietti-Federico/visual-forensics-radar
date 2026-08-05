from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from pdf_forensics.plugins.features.streams_extractor import StreamsFeatureExtractor
from tests.fixtures.feature_helpers import get_feature
from tests.fixtures.pdf_builder import PdfBuilder


def test_no_streams() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")], size=2, root_ref="1 0 R"
    )
    document = PdfDocumentParser().parse(builder.build())

    features = StreamsFeatureExtractor().extract(document)

    assert get_feature(features, "streams.stream_count").value == 0
    assert get_feature(features, "streams.filter_histogram").value == {}
    assert get_feature(features, "streams.total_raw_bytes").value == 0


def test_stream_with_unsupported_filter_is_counted() -> None:
    builder = PdfBuilder()
    off_stream = builder.add_stream_object(1, 0, "<< /Length 5 /Filter /DCTDecode >>", b"HELLO")
    off_catalog = builder.add_object(2, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off_stream, 0, "n"), (2, off_catalog, 0, "n")],
        size=3,
        root_ref="2 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = StreamsFeatureExtractor().extract(document)

    assert get_feature(features, "streams.stream_count").value == 1
    assert get_feature(features, "streams.filter_histogram").value == {"DCTDecode": 1}
    assert get_feature(features, "streams.total_raw_bytes").value == 5
    # DCTDecode on a plain content stream is never run through decode_stream()
    # (only xref/object streams are) so no anomaly is recorded here.
    assert get_feature(features, "streams.unsupported_filter_anomaly_count").value == 0


def test_object_stream_with_unsupported_filter_records_anomaly() -> None:
    builder = PdfBuilder()
    off_catalog = builder.add_object(1, 0, "<< /Type /Catalog >>")
    container_offset = builder.add_stream_object(
        3, 0, "<< /Type /ObjStm /N 1 /First 4 /Filter /DCTDecode /Length 4 >>", b"junk"
    )
    builder.add_xref_stream(
        obj_num=2,
        entries={1: (1, off_catalog, 0), 3: (1, container_offset, 0), 4: (2, 3, 0)},
        size=5,
        root_ref="1 0 R",
    )
    document = PdfDocumentParser().parse(builder.build())

    features = StreamsFeatureExtractor().extract(document)

    assert get_feature(features, "streams.unsupported_filter_anomaly_count").value == 1
