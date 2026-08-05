"""Cross-reference streams (ISO 32000-1 §7.5.8), the PDF 1.5+ alternative to classic xref tables.

A cross-reference stream is itself a normal indirect object (`/Type /XRef`) whose
decoded content is a flat sequence of fixed-width binary rows described by `/W`.
It replaces both the classic `xref` table *and* the `trailer` keyword — the
stream object's own dictionary doubles as the trailer.
"""

from __future__ import annotations

from pdf_forensics.domain.pdf.anomalies import AnomalyCollector
from pdf_forensics.domain.pdf.document import Revision, XrefEntry, XrefEntryType
from pdf_forensics.domain.pdf.objects import PdfArray, PdfNumber, PdfObject, PdfStream
from pdf_forensics.infrastructure.parsing.stream_decoding import decode_stream


def parse_xref_stream(
    stream_obj: PdfObject, offset: int, anomalies: AnomalyCollector
) -> Revision | None:
    """Parse a `/Type /XRef` stream object into a `Revision`, or `None` if it isn't one."""
    if not isinstance(stream_obj.value, PdfStream):
        return None
    stream = stream_obj.value
    if stream.dictionary.get_name("Type") != "XRef":
        return None

    widths = _int_tuple(stream.dictionary.get_array("W"))
    if widths is None or len(widths) != 3:
        return None

    size = stream.dictionary.get_int("Size") or 0
    index_pairs = _int_tuple(stream.dictionary.get_array("Index"))
    if index_pairs is None or len(index_pairs) % 2 != 0 or not index_pairs:
        index_pairs = (0, size)

    decoded = decode_stream(stream, anomalies)
    if decoded is None:
        return Revision(
            trailer=stream.dictionary, xref_entries={}, xref_offset=offset, is_xref_stream=True
        )

    row_width = sum(widths)
    entries: dict[int, XrefEntry] = {}
    pos = 0
    for pair_index in range(0, len(index_pairs), 2):
        start = index_pairs[pair_index]
        count = index_pairs[pair_index + 1]
        for i in range(count):
            if row_width == 0 or pos + row_width > len(decoded):
                break
            row = decoded[pos : pos + row_width]
            pos += row_width
            fields = _split_row(row, widths)
            entries[start + i] = _entry_from_row(start + i, fields)

    return Revision(
        trailer=stream.dictionary, xref_entries=entries, xref_offset=offset, is_xref_stream=True
    )


def _int_tuple(array: PdfArray | None) -> tuple[int, ...] | None:
    if array is None:
        return None
    values: list[int] = []
    for item in array.items:
        if not isinstance(item, PdfNumber):
            return None
        values.append(item.as_int())
    return tuple(values)


def _split_row(row: bytes, widths: tuple[int, int, int]) -> tuple[int, int, int]:
    values: list[int] = []
    pos = 0
    for index, width in enumerate(widths):
        if width == 0:
            values.append(1 if index == 0 else 0)
            continue
        values.append(int.from_bytes(row[pos : pos + width], "big"))
        pos += width
    return values[0], values[1], values[2]


def _entry_from_row(obj_num: int, fields: tuple[int, int, int]) -> XrefEntry:
    entry_type_code, field2, field3 = fields
    if entry_type_code == 0:
        return XrefEntry(
            obj_num, generation=field3, entry_type=XrefEntryType.FREE, offset_or_stream_obj_num=0
        )
    if entry_type_code == 2:
        return XrefEntry(
            obj_num,
            generation=0,
            entry_type=XrefEntryType.COMPRESSED,
            offset_or_stream_obj_num=field2,
            index_in_stream=field3,
        )
    return XrefEntry(
        obj_num,
        generation=field3,
        entry_type=XrefEntryType.IN_USE,
        offset_or_stream_obj_num=field2,
    )
