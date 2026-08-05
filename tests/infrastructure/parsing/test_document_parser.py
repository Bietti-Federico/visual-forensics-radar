import pytest

from pdf_forensics.domain.pdf.anomalies import AnomalyCode
from pdf_forensics.domain.pdf.errors import NotAPdfError, UnrecoverableStructureError
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfReference
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser
from tests.fixtures.pdf_builder import PdfBuilder


def _catalog_pages_page(builder: PdfBuilder) -> tuple[int, int, int]:
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    off3 = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R >>")
    return off1, off2, off3


def test_single_revision_classic_xref() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert {key[0] for key in document.objects} == {1, 2, 3}
    assert document.trailer.get_ref("Root") == PdfReference(1, 0)
    assert len(document.revisions) == 1
    assert document.anomalies == []
    assert document.pdf_version == "1.7"


def test_pdf_version_is_none_when_header_has_no_version_digits() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )
    data = builder.build().replace(b"%PDF-1.7", b"%PDF-broken")

    document = PdfDocumentParser().parse(data)

    assert document.pdf_version is None


def test_multi_revision_incremental_update() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    first_xref = builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n"), (3, off3, 0, "n")],
        size=4,
        root_ref="1 0 R",
    )
    off3b = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R /Rotate 90 >>")
    builder.add_classic_xref_and_trailer(
        [(3, off3b, 0, "n")], size=4, root_ref="1 0 R", prev=first_xref
    )

    document = PdfDocumentParser().parse(builder.build())

    assert len(document.revisions) == 2
    page = document.objects[(3, 0)]
    assert isinstance(page.value, PdfDictionary)
    assert page.value.get_int("Rotate") == 90  # the later revision's definition wins


def test_xref_stream_based_document() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    builder.add_xref_stream(
        obj_num=4,
        entries={1: (1, off1, 0), 2: (1, off2, 0), 3: (1, off3, 0)},
        size=5,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert {key[0] for key in document.objects} == {1, 2, 3, 4}
    assert document.trailer.get_ref("Root") == PdfReference(1, 0)
    assert document.revisions[0].is_xref_stream is True


def test_object_stream_with_three_compressed_objects() -> None:
    builder = PdfBuilder()
    off_catalog = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    objstm_offset = builder.add_object_stream(
        container_obj_num=5,
        packed_objects=[
            (2, "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>"),
            (3, "<< /Type /Page /Parent 2 0 R >>"),
            (4, "<< /Type /Page /Parent 2 0 R >>"),
        ],
    )
    builder.add_xref_stream(
        obj_num=6,
        entries={
            1: (1, off_catalog, 0),
            5: (1, objstm_offset, 0),
            2: (2, 5, 0),
            3: (2, 5, 1),
            4: (2, 5, 2),
        },
        size=7,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert {key[0] for key in document.objects} == {1, 2, 3, 4, 5, 6}
    pages = document.objects[(2, 0)].value
    assert isinstance(pages, PdfDictionary)
    assert pages.get_int("Count") == 2


def test_broken_xref_offset_is_flagged_and_recovered() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, off3 + 5, 0, "n"),  # deliberately wrong offset for object 3
        ],
        size=4,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (3, 0) in document.objects  # recovered via brute-force scan
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.XREF_OFFSET_MISMATCH in codes


def test_duplicate_object_id_in_same_xref_table_latest_wins() -> None:
    builder = PdfBuilder()
    off1, off2, off3 = _catalog_pages_page(builder)
    off3_dup = builder.add_object(3, 0, "<< /Type /Page /Parent 2 0 R /Marker (second) >>")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, off3, 0, "n"),
            (3, off3_dup, 0, "n"),
        ],
        size=4,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.DUPLICATE_OBJECT_ID in codes
    page = document.objects[(3, 0)].value
    assert isinstance(page, PdfDictionary)
    assert page.entries["Marker"].value == b"second"  # the later definition wins


def test_non_pdf_bytes_raise_not_a_pdf_error() -> None:
    with pytest.raises(NotAPdfError):
        PdfDocumentParser().parse(b"this is not a PDF file at all")


def test_header_with_no_objects_raises_unrecoverable() -> None:
    with pytest.raises(UnrecoverableStructureError):
        PdfDocumentParser().parse(b"%PDF-1.7\nthere is nothing parseable here\n")


def test_hybrid_reference_file_merges_xrefstm_entries() -> None:
    """A classic table whose trailer has /XRefStm should absorb the stream's entries."""
    builder = PdfBuilder()
    off_catalog = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off_pages = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    objstm_offset = builder.add_object_stream(
        container_obj_num=10, packed_objects=[(11, "<< /Type /Extra >>")]
    )
    xref_stream_offset = builder.add_xref_stream(
        obj_num=12,
        entries={10: (1, objstm_offset, 0), 11: (2, 10, 0)},
        size=13,
        root_ref="1 0 R",
    )
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off_catalog, 0, "n"),
            (2, off_pages, 0, "n"),
            (12, xref_stream_offset, 0, "n"),
        ],
        size=13,
        root_ref="1 0 R",
        extra_trailer=f"/XRefStm {xref_stream_offset}",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (11, 0) in document.objects  # only reachable via the hybrid /XRefStm


def test_unresolvable_object_offset_without_recovery_match_is_skipped() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    # Object 3 is referenced by the xref but its body was never written anywhere,
    # so the offset-mismatch recovery scan for "3 0 obj" cannot find it either.
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, 0, 0, "n"),
        ],
        size=4,
        root_ref="1 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert (3, 0) not in document.objects
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.XREF_OFFSET_MISMATCH in codes


def test_encrypted_document_flags_is_encrypted_via_indirect_dict() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    off_encrypt = builder.add_object(3, 0, "<< /Filter /Standard /V 1 /R 2 >>")
    builder.add_classic_xref_and_trailer(
        [
            (0, 0, 65535, "f"),
            (1, off1, 0, "n"),
            (2, off2, 0, "n"),
            (3, off_encrypt, 0, "n"),
        ],
        size=4,
        root_ref="1 0 R",
        extra_trailer="/Encrypt 3 0 R",
    )

    document = PdfDocumentParser().parse(builder.build())

    assert document.is_encrypted is True
    assert document.encryption_dict is not None
    assert document.encryption_dict.get_name("Filter") == "Standard"


def test_brute_force_recovery_finds_trailer_with_root() -> None:
    builder = PdfBuilder()
    _catalog_pages_page(builder)
    # A "trailer" block with no preceding "xref"/"startxref": the object-scan
    # fallback still runs, but the trailer (and its /Root) is separately recovered.
    tail = b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
    data = builder.build() + tail

    document = PdfDocumentParser().parse(data)

    assert document.trailer.get_ref("Root") == PdfReference(1, 0)
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.TRAILER_MISSING_ROOT not in codes


def test_brute_force_recovery_when_xref_is_entirely_missing() -> None:
    builder = PdfBuilder()
    _catalog_pages_page(builder)
    # No xref/trailer/startxref appended at all: the file ends right after the objects.
    data = builder.build()

    document = PdfDocumentParser().parse(data)

    assert {key[0] for key in document.objects} == {1, 2, 3}
    codes = [a.code for a in document.anomalies]
    assert AnomalyCode.XREF_UNPARSEABLE_FALLBACK_USED in codes
