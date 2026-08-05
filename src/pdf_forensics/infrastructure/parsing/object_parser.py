"""Recursive-descent parser from tokens to COS values (ISO 32000-1 §7.3) and objects.

Every loop here (array elements, dictionary entries) explicitly checks for its own
terminator and EOF *before* recursing, and falls back to forced single-token
progress if a nested parse makes none — a forensics parser must never hang on
adversarial or corrupted input, only degrade to recording an anomaly and moving on.
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.objects import (
    PdfArray,
    PdfBoolean,
    PdfDictionary,
    PdfHexString,
    PdfLiteralString,
    PdfName,
    PdfNull,
    PdfNumber,
    PdfObject,
    PdfReference,
    PdfStream,
    PdfValue,
)
from pdf_forensics.infrastructure.parsing.tokenizer import (
    WHITESPACE_BYTES,
    PdfTokenizer,
    Token,
    TokenKind,
)

#: Resolves an indirect `/Length` reference to a byte count, or `None` if it can't
#: be resolved yet (e.g. the referenced object hasn't been parsed). Injected by
#: `document_parser.py` once an xref is available; `None` means "not resolvable".
LengthResolver = Callable[[PdfReference], "int | None"]

_ENDSTREAM = b"endstream"


class ObjectParser:
    def __init__(
        self,
        tokenizer: PdfTokenizer,
        anomalies: AnomalyCollector,
        length_resolver: LengthResolver | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._anomalies = anomalies
        self._length_resolver = length_resolver

    def parse_value(self, pos: int) -> tuple[PdfValue, int]:
        """Parse one COS value starting at `pos`, returning `(value, next_pos)`."""
        token = self._tokenizer.next_token(pos)
        return self._parse_value_from_token(token)

    def parse_indirect_object(self, offset: int) -> tuple[PdfObject | None, int]:
        """Parse `N G obj ... endobj` at `offset`.

        Returns `(None, offset + 1)` on failure to match the expected shape, so a
        caller scanning for objects can always make forward progress.
        """
        tok_num = self._tokenizer.next_token(offset)
        obj_num = self._token_int(tok_num)
        if obj_num is None:
            return None, offset + 1

        tok_gen = self._tokenizer.next_token(tok_num.end)
        generation = self._token_int(tok_gen)
        if generation is None:
            return None, offset + 1

        tok_obj = self._tokenizer.next_token(tok_gen.end)
        if not self._is_keyword(tok_obj, b"obj"):
            return None, offset + 1

        value, pos = self.parse_value(tok_obj.end)

        endobj_check = self._tokenizer.next_token(pos)
        if self._is_keyword(endobj_check, b"endobj"):
            pos = endobj_check.end

        return PdfObject(obj_num=obj_num, generation=generation, value=value, offset=offset), pos

    def _parse_value_from_token(self, token: Token) -> tuple[PdfValue, int]:
        if token.kind == TokenKind.NUMBER:
            return self._parse_number_or_reference(token)
        if token.kind == TokenKind.NAME:
            assert isinstance(token.value, str)
            return PdfName(token.value), token.end
        if token.kind == TokenKind.LITERAL_STRING:
            assert isinstance(token.value, bytes)
            return PdfLiteralString(token.value), token.end
        if token.kind == TokenKind.HEX_STRING:
            assert isinstance(token.value, bytes)
            return PdfHexString(token.value), token.end
        if token.kind == TokenKind.ARRAY_START:
            return self._parse_array(token.end)
        if token.kind == TokenKind.DICT_START:
            return self._parse_dict_or_stream(token.end)
        if token.kind == TokenKind.KEYWORD:
            if token.value == b"true":
                return PdfBoolean(True), token.end
            if token.value == b"false":
                return PdfBoolean(False), token.end
            if token.value == b"null":
                return PdfNull(), token.end
            return PdfNull(), token.end
        # ARRAY_END / DICT_END / EOF / UNKNOWN encountered where a value was
        # expected: report an empty value without consuming, so the caller's
        # own terminator/EOF check (not this method) decides how to proceed.
        return PdfNull(), token.start

    def _parse_number_or_reference(self, first: Token) -> tuple[PdfValue, int]:
        second = self._tokenizer.next_token(first.end)
        if second.kind == TokenKind.NUMBER:
            third = self._tokenizer.next_token(second.end)
            if self._is_keyword(third, b"R"):
                obj_num = self._token_int(first)
                generation = self._token_int(second)
                if obj_num is not None and generation is not None:
                    return PdfReference(obj_num, generation), third.end
        return self._number_from_token(first), first.end

    def _parse_array(self, pos: int) -> tuple[PdfArray, int]:
        items: list[PdfValue] = []
        while True:
            token = self._tokenizer.next_token(pos)
            if token.kind in (TokenKind.ARRAY_END, TokenKind.EOF):
                pos = token.end
                break
            value, new_pos = self.parse_value(pos)
            items.append(value)
            pos = new_pos if new_pos > pos else token.end
        return PdfArray(tuple(items)), pos

    def _parse_dict_or_stream(self, pos: int) -> tuple[PdfValue, int]:
        entries: dict[str, PdfValue] = {}
        while True:
            token = self._tokenizer.next_token(pos)
            if token.kind in (TokenKind.DICT_END, TokenKind.EOF):
                pos = token.end
                break
            if token.kind != TokenKind.NAME:
                pos = token.end
                continue
            assert isinstance(token.value, str)
            key = token.value
            value, new_pos = self.parse_value(token.end)
            entries[key] = value
            pos = new_pos if new_pos > token.end else token.end

        dictionary = PdfDictionary(entries)
        stream_check = self._tokenizer.next_token(pos)
        if self._is_keyword(stream_check, b"stream"):
            return self._read_stream(dictionary, stream_check.end)
        return dictionary, pos

    def _read_stream(self, dictionary: PdfDictionary, pos: int) -> tuple[PdfStream, int]:
        data = self._tokenizer.data
        length = len(data)

        if pos < length and data[pos] == 0x0D:
            pos += 1
            if pos < length and data[pos] == 0x0A:
                pos += 1
        elif pos < length and data[pos] == 0x0A:
            pos += 1

        stream_start = pos
        declared_length = self._resolve_declared_length(dictionary)

        if declared_length is not None:
            candidate_end = stream_start + declared_length
            if 0 <= candidate_end <= length:
                check_pos = candidate_end
                while check_pos < length and data[check_pos] in WHITESPACE_BYTES:
                    check_pos += 1
                if data[check_pos : check_pos + len(_ENDSTREAM)] == _ENDSTREAM:
                    raw = data[stream_start:candidate_end]
                    return PdfStream(dictionary, raw), check_pos + len(_ENDSTREAM)

        self._anomalies.record(
            AnomalyCode.STREAM_LENGTH_MISMATCH,
            AnomalySeverity.WARNING,
            "Stream /Length did not match the position of 'endstream'; recovered by scanning.",
            byte_offset=stream_start,
        )
        return self._scan_for_endstream(dictionary, stream_start)

    def _resolve_declared_length(self, dictionary: PdfDictionary) -> int | None:
        declared = dictionary.get_int("Length")
        if declared is not None:
            return declared
        length_ref = dictionary.get_ref("Length")
        if length_ref is not None and self._length_resolver is not None:
            return self._length_resolver(length_ref)
        return None

    def _scan_for_endstream(
        self, dictionary: PdfDictionary, stream_start: int
    ) -> tuple[PdfStream, int]:
        data = self._tokenizer.data
        idx = data.find(_ENDSTREAM, stream_start)
        if idx == -1:
            return PdfStream(dictionary, data[stream_start:]), len(data)
        raw = data[stream_start:idx]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        elif raw.endswith(b"\n") or raw.endswith(b"\r"):
            raw = raw[:-1]
        return PdfStream(dictionary, raw), idx + len(_ENDSTREAM)

    @staticmethod
    def _is_keyword(token: Token, keyword: bytes) -> bool:
        return token.kind == TokenKind.KEYWORD and token.value == keyword

    def _number_from_token(self, token: Token) -> PdfNumber:
        assert isinstance(token.value, bytes)
        text = token.value.decode("ascii", errors="replace")
        try:
            value = Decimal(text)
        except InvalidOperation:
            value = Decimal(0)
        return PdfNumber(value=value, is_integer=b"." not in token.value)

    def _token_int(self, token: Token) -> int | None:
        if token.kind != TokenKind.NUMBER:
            return None
        number = self._number_from_token(token)
        try:
            return number.as_int()
        except (ValueError, ArithmeticError):
            return None
