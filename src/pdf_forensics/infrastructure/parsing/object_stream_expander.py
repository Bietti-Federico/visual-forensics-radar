"""Object streams (ISO 32000-1 §7.5.7): compressed objects packed inside one stream object.

Objects inside an object stream cannot themselves be streams (ISO 32000-1 §7.5.7,
Note 2), so a single `ObjectParser` bound to the decoded content can safely read
every packed object's value without ever needing to look for `stream`/`endstream`
inside it.
"""

from __future__ import annotations

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.objects import PdfObject, PdfStream
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.stream_decoding import decode_stream
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer, TokenKind


def expand_object_stream(stream_obj: PdfObject, anomalies: AnomalyCollector) -> list[PdfObject]:
    """Return every object packed inside `stream_obj`, or `[]` if it isn't an object stream."""
    if not isinstance(stream_obj.value, PdfStream):
        return []
    stream = stream_obj.value
    if stream.dictionary.get_name("Type") != "ObjStm":
        return []

    count = stream.dictionary.get_int("N")
    first_offset = stream.dictionary.get_int("First")
    if count is None or first_offset is None:
        return []

    decoded = decode_stream(stream, anomalies)
    if decoded is None:
        return []

    body_tokenizer = PdfTokenizer(decoded)
    header_pos = 0
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        num_tok = body_tokenizer.next_token(header_pos)
        off_tok = body_tokenizer.next_token(num_tok.end)
        header_pos = off_tok.end
        if num_tok.kind != TokenKind.NUMBER or off_tok.kind != TokenKind.NUMBER:
            anomalies.record(
                AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE,
                AnomalySeverity.WARNING,
                f"Object stream {stream_obj.obj_num} header ended before all /N pairs were read.",
            )
            break
        obj_num = _parse_int(num_tok.value)
        data_offset = _parse_int(off_tok.value)
        if obj_num is None or data_offset is None:
            continue
        pairs.append((obj_num, data_offset))

    object_parser = ObjectParser(body_tokenizer, anomalies)
    objects: list[PdfObject] = []
    for obj_num, data_offset in pairs:
        absolute_offset = first_offset + data_offset
        if not 0 <= absolute_offset < len(decoded):
            anomalies.record(
                AnomalyCode.OBJSTM_INDEX_OUT_OF_RANGE,
                AnomalySeverity.WARNING,
                f"Object {obj_num} offset {absolute_offset} is outside the decoded stream "
                f"(length {len(decoded)}).",
            )
            continue
        value, _ = object_parser.parse_value(absolute_offset)
        objects.append(PdfObject(obj_num=obj_num, generation=0, value=value, offset=None))
    return objects


def _parse_int(raw: bytes | str | None) -> int | None:
    if not isinstance(raw, bytes):
        return None
    try:
        return int(raw.decode("ascii", errors="replace"))
    except ValueError:
        return None
