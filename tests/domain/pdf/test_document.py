from decimal import Decimal

from pdf_forensics.domain.pdf.document import PdfDocument, Revision, XrefEntry, XrefEntryType
from pdf_forensics.domain.pdf.objects import (
    PdfDictionary,
    PdfName,
    PdfNumber,
    PdfObject,
    PdfReference,
)


def _revision(trailer_entries: dict, xref_entries: dict[int, XrefEntry] | None = None) -> Revision:
    return Revision(
        trailer=PdfDictionary(trailer_entries),
        xref_entries=xref_entries or {},
        xref_offset=0,
        is_xref_stream=False,
    )


def test_trailer_merges_older_revisions_filling_gaps() -> None:
    newest = _revision({"Size": PdfNumber(Decimal(2), True)})
    oldest = _revision({"Root": PdfReference(1, 0), "Size": PdfNumber(Decimal(1), True)})
    document = PdfDocument(revisions=[newest, oldest], objects={})

    merged = document.trailer
    assert merged.get_int("Size") == 2  # newest wins on conflict
    assert merged.get_ref("Root") == PdfReference(1, 0)  # inherited from the older revision


def test_latest_revision_is_first_in_list() -> None:
    newest = _revision({"Marker": PdfName("newest")})
    oldest = _revision({"Marker": PdfName("oldest")})
    document = PdfDocument(revisions=[newest, oldest], objects={})
    assert document.latest_revision is newest


def test_resolve_looks_up_object_by_reference() -> None:
    obj = PdfObject(obj_num=1, generation=0, value=PdfName("Catalog"), offset=0)
    document = PdfDocument(revisions=[_revision({})], objects={(1, 0): obj})
    assert document.resolve(PdfReference(1, 0)) is obj
    assert document.resolve(PdfReference(2, 0)) is None


def test_xref_entry_type_enum_values() -> None:
    assert XrefEntryType.FREE.value == "free"
    assert XrefEntryType.IN_USE.value == "in_use"
    assert XrefEntryType.COMPRESSED.value == "compressed"
