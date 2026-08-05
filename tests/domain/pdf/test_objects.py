from decimal import Decimal

from pdf_forensics.domain.pdf.objects import (
    PdfArray,
    PdfDictionary,
    PdfHexString,
    PdfLiteralString,
    PdfName,
    PdfNumber,
    PdfReference,
    PdfStream,
)


def test_pdf_number_as_int_truncates_decimal() -> None:
    assert PdfNumber(Decimal("3.9"), is_integer=False).as_int() == 3


def test_literal_string_decodes_latin1_by_default() -> None:
    assert PdfLiteralString(b"hello").decode_text() == "hello"


def test_hex_string_decodes_utf16be_with_bom() -> None:
    raw = "hola".encode("utf-16-be")
    assert PdfHexString(b"\xfe\xff" + raw).decode_text() == "hola"


def test_hex_string_falls_back_to_latin1_on_invalid_utf16() -> None:
    # BOM present, but an odd number of trailing bytes can't form UTF-16 code units.
    broken = b"\xfe\xff\x41"
    assert PdfHexString(broken).decode_text() == broken.decode("latin-1")


def test_dictionary_accessors_return_none_on_type_mismatch() -> None:
    d = PdfDictionary(
        {
            "Name": PdfName("Foo"),
            "Num": PdfNumber(Decimal(5), is_integer=True),
            "Ref": PdfReference(1, 0),
            "Arr": PdfArray((PdfNumber(Decimal(1), True),)),
        }
    )
    assert d.get_name("Name") == "Foo"
    assert d.get_name("Num") is None
    assert d.get_int("Num") == 5
    assert d.get_int("Name") is None
    assert d.get_ref("Ref") == PdfReference(1, 0)
    assert d.get_ref("Name") is None
    assert d.get_array("Arr") is not None
    assert d.get_array("Ref") is None
    assert d.get_dict("Name") is None
    assert d.get("Missing") is None


def test_stream_filter_names_from_single_name() -> None:
    stream = PdfStream(PdfDictionary({"Filter": PdfName("FlateDecode")}), b"")
    assert stream.filter_names == ("FlateDecode",)


def test_stream_filter_names_from_array() -> None:
    stream = PdfStream(
        PdfDictionary({"Filter": PdfArray((PdfName("ASCII85Decode"), PdfName("FlateDecode")))}),
        b"",
    )
    assert stream.filter_names == ("ASCII85Decode", "FlateDecode")


def test_stream_filter_names_empty_when_absent() -> None:
    stream = PdfStream(PdfDictionary({}), b"")
    assert stream.filter_names == ()
