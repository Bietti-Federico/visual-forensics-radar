from pdf_forensics.domain.pdf.anomalies import AnomalyCollector
from pdf_forensics.domain.pdf.document import XrefEntryType
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer
from pdf_forensics.infrastructure.parsing.xref_stream_parser import parse_xref_stream
from tests.fixtures.pdf_builder import PdfBuilder


def _parse_xref_stream_at(data: bytes, offset: int):
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    obj, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert obj is not None
    return parse_xref_stream(obj, offset, anomalies), anomalies


def test_parses_xref_stream_entry_types() -> None:
    builder = PdfBuilder()
    offset = builder.add_xref_stream(
        obj_num=1,
        entries={
            2: (1, 50, 0),
            4: (2, 5, 3),
        },  # obj2 in-use@50, obj4 compressed in container 5 idx 3
        size=5,
        root_ref="2 0 R",
    )
    data = builder.build()

    revision, anomalies = _parse_xref_stream_at(data, offset)
    assert revision is not None
    assert revision.is_xref_stream is True
    assert revision.xref_entries[1].entry_type == XrefEntryType.IN_USE
    assert revision.xref_entries[1].offset_or_stream_obj_num == offset
    assert revision.xref_entries[2].offset_or_stream_obj_num == 50
    assert revision.xref_entries[3].entry_type == XrefEntryType.FREE
    assert revision.xref_entries[4].entry_type == XrefEntryType.COMPRESSED
    assert revision.xref_entries[4].offset_or_stream_obj_num == 5
    assert revision.xref_entries[4].index_in_stream == 3
    assert revision.trailer.get_int("Size") == 5
    assert not anomalies.to_list()


def test_uncompressed_xref_stream() -> None:
    builder = PdfBuilder()
    offset = builder.add_xref_stream(
        obj_num=1, entries={2: (1, 30, 0)}, size=3, root_ref="2 0 R", compress=False
    )
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is not None
    assert revision.xref_entries[2].offset_or_stream_obj_num == 30


def test_not_an_xref_stream_returns_none() -> None:
    builder = PdfBuilder()
    offset = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    obj, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert obj is not None
    assert parse_xref_stream(obj, offset, anomalies) is None


def test_stream_object_of_wrong_type_returns_none() -> None:
    """A stream whose /Type isn't /XRef (e.g. an /ObjStm) is not a cross-reference stream."""
    builder = PdfBuilder()
    offset = builder.add_object_stream(container_obj_num=1, packed_objects=[(2, "/Foo")])
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is None


def test_missing_w_array_returns_none() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(1, 0, "<< /Type /XRef /Size 2 /Root 2 0 R /Length 0 >>", b"")
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is None


def test_undecodable_stream_yields_empty_entries() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1,
        0,
        "<< /Type /XRef /Size 2 /W [1 2 1] /Root 2 0 R /Filter /FlateDecode /Length 4 >>",
        b"nope",
    )
    revision, anomalies = _parse_xref_stream_at(builder.build(), offset)
    assert revision is not None
    assert revision.xref_entries == {}
    assert anomalies.to_list()


def test_truncated_row_data_stops_without_crashing() -> None:
    # /W declares 4-byte rows but only 2 bytes of decoded content are provided.
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1,
        0,
        "<< /Type /XRef /Size 2 /W [1 2 1] /Root 2 0 R /Length 2 >>",
        b"\x01\x00",
    )
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is not None
    assert revision.xref_entries == {}


def test_zero_width_fields_default_type_one_and_zero() -> None:
    # /W [0 1 1]: field 1 (type) has width 0 and defaults to 1 (in-use); field 2 has
    # a real width; object 0 (also width-0 type) defaults its type to 1 as well, so
    # both rows are read from a 2-byte-per-row buffer.
    row0 = bytes([10, 0])  # field2=10, field3=0
    row1 = bytes([20, 5])  # field2=20, field3=5
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1,
        0,
        "<< /Type /XRef /Size 2 /W [0 1 1] /Root 2 0 R /Length 4 >>",
        row0 + row1,
    )
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is not None
    assert revision.xref_entries[0].entry_type == XrefEntryType.IN_USE
    assert revision.xref_entries[0].offset_or_stream_obj_num == 10
    assert revision.xref_entries[1].offset_or_stream_obj_num == 20


def test_explicit_index_array_selects_object_numbers() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1,
        0,
        "<< /Type /XRef /Size 20 /W [1 2 1] /Index [5 1] /Root 2 0 R /Length 4 >>",
        bytes([1, 0, 42, 0]),
    )
    revision, _ = _parse_xref_stream_at(builder.build(), offset)
    assert revision is not None
    assert list(revision.xref_entries) == [5]
    assert revision.xref_entries[5].offset_or_stream_obj_num == 42
