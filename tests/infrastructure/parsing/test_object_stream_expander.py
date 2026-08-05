from decimal import Decimal

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfName, PdfNumber
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.object_stream_expander import (
    _parse_int,
    expand_object_stream,
)
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer
from tests.fixtures.pdf_builder import PdfBuilder


def test_expands_packed_objects() -> None:
    builder = PdfBuilder()
    offset = builder.add_object_stream(
        container_obj_num=10,
        packed_objects=[
            (5, "<< /Type /Page >>"),
            (6, "42"),
        ],
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None

    objects = expand_object_stream(container, anomalies)
    by_num = {obj.obj_num: obj for obj in objects}
    assert by_num[5].generation == 0
    assert by_num[5].value == PdfDictionary({"Type": PdfName("Page")})
    assert by_num[6].value == PdfNumber(Decimal(42), True)
    assert not anomalies.to_list()


def test_uncompressed_object_stream() -> None:
    builder = PdfBuilder()
    offset = builder.add_object_stream(
        container_obj_num=10, packed_objects=[(1, "/Foo")], compress=False
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    objects = expand_object_stream(container, anomalies)
    assert objects[0].value == PdfName("Foo")


def test_not_an_object_stream_returns_empty() -> None:
    builder = PdfBuilder()
    offset = builder.add_object(1, 0, "<< /Type /Catalog >>")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    assert expand_object_stream(container, anomalies) == []


def test_missing_first_returns_empty() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(1, 0, "<< /Type /ObjStm /N 1 >>", b"5 0 /Foo")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    assert expand_object_stream(container, anomalies) == []


def test_undecodable_stream_returns_empty() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1, 0, "<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode /Length 4 >>", b"nope"
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    assert expand_object_stream(container, anomalies) == []


def test_header_shorter_than_declared_n_records_anomaly() -> None:
    # /N claims 3 pairs but only one is actually present in the header.
    builder = PdfBuilder()
    offset = builder.add_stream_object(1, 0, "<< /Type /ObjStm /N 3 /First 4 >>", b"5 0 /Foo")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    objects = expand_object_stream(container, anomalies)
    assert [obj.obj_num for obj in objects] == [5]
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE in codes


def test_offset_outside_decoded_stream_records_anomaly() -> None:
    builder = PdfBuilder()
    offset = builder.add_stream_object(1, 0, "<< /Type /ObjStm /N 1 /First 4 >>", b"5 999")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    objects = expand_object_stream(container, anomalies)
    assert objects == []
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE in codes


def test_parse_int_rejects_non_bytes() -> None:
    assert _parse_int(None) is None
    assert _parse_int("5") is None


def test_stream_of_wrong_type_returns_empty() -> None:
    """A stream whose /Type isn't /ObjStm (e.g. an /XRef stream) is not an object stream."""
    builder = PdfBuilder()
    offset = builder.add_xref_stream(obj_num=1, entries={}, size=2, root_ref="1 0 R")
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    assert expand_object_stream(container, anomalies) == []


def test_non_integer_header_pair_is_skipped_silently() -> None:
    # The first pair's offset field ("3.0") isn't a plain integer, so it's skipped;
    # the second pair is still read normally.
    builder = PdfBuilder()
    offset = builder.add_stream_object(
        1, 0, "<< /Type /ObjStm /N 2 /First 12 >>", b"5 3.0 6 0   /Foo(second)"
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(builder.build())
    container, _ = ObjectParser(tokenizer, anomalies).parse_indirect_object(offset)
    assert container is not None
    objects = expand_object_stream(container, anomalies)
    assert [obj.obj_num for obj in objects] == [6]
