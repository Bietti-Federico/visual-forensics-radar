from pdf_forensics.domain.pdf.anomalies import AnomalyCode
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfReference, PdfStream
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser, _read_int
from pdf_forensics.infrastructure.parsing.tokenizer import Token, TokenKind
from tests.fixtures.pdf_builder import PdfBuilder


def test_startxref_pointing_at_garbage_falls_back_to_brute_force() -> None:
    builder = PdfBuilder()
    builder.add_object(1, 0, "<< /Type /Catalog >>")
    garbage_offset = builder.add_raw(b"this is neither an xref section nor an object\n")
    builder.add_raw(f"startxref\n{garbage_offset}\n%%EOF\n".encode("ascii"))

    document = PdfDocumentParser().parse(builder.build())

    assert (1, 0) in document.objects
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.XREF_UNPARSEABLE_FALLBACK_USED in codes


def test_hybrid_xrefstm_pointing_nowhere_is_ignored() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")],
        size=2,
        root_ref="1 0 R",
        extra_trailer="/XRefStm 999999",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (1, 0) in document.objects
    assert document.trailer.get_ref("Root") == PdfReference(1, 0)


def test_hybrid_xrefstm_pointing_at_non_xref_object_is_ignored() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")],
        size=2,
        root_ref="1 0 R",
        extra_trailer=f"/XRefStm {off1}",  # points at the Catalog object, not an XRef stream
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (1, 0) in document.objects


def test_brute_force_duplicate_object_definition() -> None:
    builder = PdfBuilder()
    builder.add_object(5, 0, "<< /Marker (first) >>")
    builder.add_object(5, 0, "<< /Marker (second) >>")

    document = PdfDocumentParser().parse(builder.build())

    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.DUPLICATE_OBJECT_ID in codes
    page = document.objects[(5, 0)].value
    assert isinstance(page, PdfDictionary)
    assert page.entries["Marker"].value == b"second"


def test_brute_force_recovered_trailer_missing_root_is_flagged() -> None:
    builder = PdfBuilder()
    builder.add_object(1, 0, "<< /Type /Catalog >>")
    data = builder.build() + b"trailer\n<< /Size 1 >>\n"

    document = PdfDocumentParser().parse(data)

    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.TRAILER_MISSING_ROOT in codes
    assert document.trailer.get_ref("Root") is None


def test_indirect_stream_length_is_resolved_end_to_end() -> None:
    builder = PdfBuilder()
    off_length = builder.add_object(1, 0, "5")
    off_stream = builder.add_stream_object(2, 0, "<< /Length 1 0 R >>", b"HELLO")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off_length, 0, "n"), (2, off_stream, 0, "n")],
        size=3,
        root_ref="2 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    stream = document.objects[(2, 0)].value
    assert isinstance(stream, PdfStream)
    assert stream.raw_data == b"HELLO"


def test_compressed_entry_with_unresolvable_container_is_skipped() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_xref_stream(
        obj_num=2,
        entries={1: (1, off1, 0), 3: (2, 999, 5)},  # container 999 was never defined
        size=4,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (3, 0) not in document.objects
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE in codes


def test_dangling_indirect_encrypt_reference_leaves_dict_unset() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")],
        size=2,
        root_ref="1 0 R",
        extra_trailer="/Encrypt 9 0 R",  # object 9 doesn't exist
    )

    document = PdfDocumentParser().parse(builder.build())

    assert document.is_encrypted is True
    assert document.encryption_dict is None


def test_inline_encrypt_dictionary_in_trailer() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n")],
        size=2,
        root_ref="1 0 R",
        extra_trailer="/Encrypt << /Filter /Standard >>",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert document.is_encrypted is True
    assert document.encryption_dict is not None
    assert document.encryption_dict.get_name("Filter") == "Standard"


def test_read_int_rejects_non_number_token() -> None:
    assert _read_int(Token(TokenKind.KEYWORD, b"foo", 0, 3)) is None


def test_read_int_rejects_undecimal_number_token() -> None:
    assert _read_int(Token(TokenKind.NUMBER, b"NaN", 0, 3)) is None
