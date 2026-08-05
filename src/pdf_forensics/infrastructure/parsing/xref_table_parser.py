"""Classic cross-reference tables (ISO 32000-1 §7.5.4): the pre-PDF-1.5 xref format.

Entries are parsed token-by-token rather than by fixed 20-byte offsets — real
files routinely deviate from the spec's exact byte-width (missing padding,
`\\n`-only line endings), and tokenizing tolerates all of that for free without
separate special-casing.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.document import Revision, XrefEntry, XrefEntryType
from pdf_forensics.domain.pdf.objects import PdfDictionary
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer, Token, TokenKind


def parse_xref_table(
    tokenizer: PdfTokenizer,
    object_parser: ObjectParser,
    offset: int,
    anomalies: AnomalyCollector,
) -> Revision | None:
    """Parse `xref ... trailer << ... >>` at `offset`, or return `None` if it isn't one."""
    xref_keyword = tokenizer.next_token(offset)
    if not (xref_keyword.kind == TokenKind.KEYWORD and xref_keyword.value == b"xref"):
        return None

    pos = xref_keyword.end
    entries: dict[int, XrefEntry] = {}

    while True:
        start_tok = tokenizer.next_token(pos)
        if start_tok.kind != TokenKind.NUMBER:
            break
        count_tok = tokenizer.next_token(start_tok.end)
        if count_tok.kind != TokenKind.NUMBER:
            break
        start = _read_int(start_tok)
        count = _read_int(count_tok)
        pos = count_tok.end
        if start is None or count is None:
            break

        for i in range(count):
            offset_tok = tokenizer.next_token(pos)
            if offset_tok.kind != TokenKind.NUMBER:
                break
            gen_tok = tokenizer.next_token(offset_tok.end)
            type_tok = tokenizer.next_token(gen_tok.end)
            pos = type_tok.end

            entry_offset = _read_int(offset_tok)
            entry_gen = _read_int(gen_tok)
            if entry_offset is None or entry_gen is None:
                continue

            obj_num = start + i
            is_free = type_tok.kind == TokenKind.KEYWORD and type_tok.value == b"f"
            if obj_num in entries:
                anomalies.record(
                    AnomalyCode.DUPLICATE_OBJECT_ID,
                    AnomalySeverity.WARNING,
                    f"Object {obj_num} has more than one entry in the same xref table; "
                    "the later entry wins.",
                    byte_offset=offset_tok.start,
                )
            entries[obj_num] = XrefEntry(
                obj_num=obj_num,
                generation=entry_gen,
                entry_type=XrefEntryType.FREE if is_free else XrefEntryType.IN_USE,
                offset_or_stream_obj_num=0 if is_free else entry_offset,
            )

    trailer_keyword = tokenizer.next_token(pos)
    if trailer_keyword.kind == TokenKind.KEYWORD and trailer_keyword.value == b"trailer":
        value, _ = object_parser.parse_value(trailer_keyword.end)
        trailer = value if isinstance(value, PdfDictionary) else PdfDictionary({})
    else:
        trailer = PdfDictionary({})

    return Revision(trailer=trailer, xref_entries=entries, xref_offset=offset, is_xref_stream=False)


def _read_int(token: Token) -> int | None:
    if token.kind != TokenKind.NUMBER or not isinstance(token.value, bytes):
        return None
    try:
        return int(Decimal(token.value.decode("ascii", errors="replace")))
    except (InvalidOperation, ValueError):
        return None
