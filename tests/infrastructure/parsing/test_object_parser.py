from decimal import Decimal

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector
from pdf_forensics.domain.pdf.objects import (
    PdfArray,
    PdfBoolean,
    PdfDictionary,
    PdfHexString,
    PdfLiteralString,
    PdfName,
    PdfNull,
    PdfNumber,
    PdfReference,
    PdfStream,
)
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer, Token, TokenKind


def _parser(data: bytes) -> ObjectParser:
    return ObjectParser(PdfTokenizer(data), AnomalyCollector())


def test_parses_boolean_and_null() -> None:
    assert _parser(b"true").parse_value(0)[0] == PdfBoolean(True)
    assert _parser(b"false").parse_value(0)[0] == PdfBoolean(False)
    assert _parser(b"null").parse_value(0)[0] == PdfNull()


def test_parses_number() -> None:
    value, pos = _parser(b"3.14").parse_value(0)
    assert value == PdfNumber(Decimal("3.14"), is_integer=False)
    assert pos == 4


def test_parses_name_literal_and_hex_string() -> None:
    assert _parser(b"/Foo").parse_value(0)[0] == PdfName("Foo")
    assert _parser(b"(bar)").parse_value(0)[0] == PdfLiteralString(b"bar")
    assert _parser(b"<FF00>").parse_value(0)[0] == PdfHexString(b"\xff\x00")


def test_parses_indirect_reference_not_two_plain_numbers() -> None:
    value, pos = _parser(b"12 0 R").parse_value(0)
    assert value == PdfReference(12, 0)
    assert pos == 6


def test_two_numbers_without_r_are_not_a_reference() -> None:
    value, pos = _parser(b"12 0").parse_value(0)
    assert value == PdfNumber(Decimal(12), is_integer=True)
    assert pos == 2


def test_parses_array_of_mixed_values() -> None:
    value, _ = _parser(b"[1 /Two (three) 4 0 R]").parse_value(0)
    assert value == PdfArray(
        (
            PdfNumber(Decimal(1), True),
            PdfName("Two"),
            PdfLiteralString(b"three"),
            PdfReference(4, 0),
        )
    )


def test_parses_dictionary() -> None:
    value, _ = _parser(b"<< /Type /Catalog /Count 3 >>").parse_value(0)
    assert value == PdfDictionary(
        {"Type": PdfName("Catalog"), "Count": PdfNumber(Decimal(3), True)}
    )


def test_unterminated_array_does_not_hang() -> None:
    value, pos = _parser(b"[1 2 3").parse_value(0)
    assert value == PdfArray(
        (PdfNumber(Decimal(1), True), PdfNumber(Decimal(2), True), PdfNumber(Decimal(3), True))
    )
    assert pos == 6


def test_stream_with_explicit_length() -> None:
    data = b"<< /Length 5 >>\nstream\nHELLO\nendstream"
    value, pos = _parser(data).parse_value(0)
    assert isinstance(value, PdfStream)
    assert value.raw_data == b"HELLO"
    assert data[pos - len(b"endstream") : pos] == b"endstream"


def test_stream_with_mismatched_length_falls_back_to_scanning() -> None:
    anomalies = AnomalyCollector()
    data = b"<< /Length 999 >>\nstream\nHELLO\nendstream"
    parser = ObjectParser(PdfTokenizer(data), anomalies)
    value, _ = parser.parse_value(0)
    assert isinstance(value, PdfStream)
    assert value.raw_data == b"HELLO"
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.STREAM_LENGTH_MISMATCH in codes


def test_stream_length_resolved_indirectly() -> None:
    def resolver(ref: PdfReference) -> int | None:
        assert ref == PdfReference(9, 0)
        return 5

    data = b"<< /Length 9 0 R >>\nstream\nHELLO\nendstream"
    parser = ObjectParser(PdfTokenizer(data), AnomalyCollector(), length_resolver=resolver)
    value, _ = parser.parse_value(0)
    assert isinstance(value, PdfStream)
    assert value.raw_data == b"HELLO"


def test_parse_indirect_object_round_trip() -> None:
    data = b"7 0 obj\n<< /Type /Page >>\nendobj\n"
    obj, pos = _parser(data).parse_indirect_object(0)
    assert obj is not None
    assert obj.obj_num == 7
    assert obj.generation == 0
    assert obj.value == PdfDictionary({"Type": PdfName("Page")})
    assert data[:pos].endswith(b"endobj")


def test_parse_indirect_object_returns_none_on_bad_shape() -> None:
    obj, pos = _parser(b"not an object").parse_indirect_object(0)
    assert obj is None
    assert pos == 1


def test_parse_indirect_object_fails_on_missing_generation() -> None:
    obj, pos = _parser(b"7 notanumber obj").parse_indirect_object(0)
    assert obj is None
    assert pos == 1


def test_parse_indirect_object_fails_on_missing_obj_keyword() -> None:
    obj, pos = _parser(b"7 0 notobj").parse_indirect_object(0)
    assert obj is None
    assert pos == 1


def test_dict_skips_non_name_token_where_key_expected() -> None:
    value, _ = _parser(b"<< 123 /Foo /Bar >>").parse_value(0)
    assert value == PdfDictionary({"Foo": PdfName("Bar")})


def test_stream_cr_only_line_ending_after_stream_keyword() -> None:
    data = b"<< /Length 5 >>\rstream\rHELLO\rendstream"
    value, _ = _parser(data).parse_value(0)
    assert isinstance(value, PdfStream)
    assert value.raw_data == b"HELLO"


def test_unterminated_stream_consumes_to_end_of_data() -> None:
    data = b"<< /Length 999 >>\nstream\nHELLO NO ENDSTREAM HERE"
    value, pos = _parser(data).parse_value(0)
    assert isinstance(value, PdfStream)
    assert pos == len(data)


def test_scan_for_endstream_trims_trailing_crlf() -> None:
    data = b"<< /Length 999 >>\nstream\nHELLO\r\nendstream"
    value, _ = _parser(data).parse_value(0)
    assert isinstance(value, PdfStream)
    assert value.raw_data == b"HELLO"


def test_number_from_token_falls_back_to_zero_on_invalid_decimal() -> None:
    parser = _parser(b"")
    bad_token = Token(TokenKind.NUMBER, b"--5", 0, 3)
    number = parser._number_from_token(bad_token)
    assert number.value == 0


def test_token_int_returns_none_when_decimal_has_no_integer_value() -> None:
    parser = _parser(b"")
    nan_token = Token(TokenKind.NUMBER, b"NaN", 0, 3)
    assert parser._token_int(nan_token) is None
