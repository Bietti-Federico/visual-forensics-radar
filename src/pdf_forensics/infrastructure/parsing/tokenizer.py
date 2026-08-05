"""Byte-level lexer for PDF COS syntax (ISO 32000-1 §7.2, Annex A).

Deliberately pull-based (`next_token(offset) -> Token`) rather than a generator:
callers (chiefly `object_parser.py`) need to stop tokenizing mid-object — e.g. the
moment a `stream` keyword is seen, raw bytes must be read directly from `data`
using the declared `/Length`, not tokenized as if they were more COS syntax. A
generator would force awkward `send()`/lookahead-buffering to support that; an
explicit offset in, token (with its own `.end` offset) out keeps control in the
caller's hands.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

_WHITESPACE = frozenset(b"\x00\t\n\x0c\r ")
_DELIMITERS = frozenset(b"()<>[]{}/%")
_NUMBER_LEAD = frozenset(b"+-.0123456789")
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")

#: Exposed for callers (e.g. the object parser reading raw stream bytes) that need
#: the same whitespace definition without duplicating it.
WHITESPACE_BYTES = _WHITESPACE

_ESCAPE_MAP = {
    ord("n"): 0x0A,
    ord("r"): 0x0D,
    ord("t"): 0x09,
    ord("b"): 0x08,
    ord("f"): 0x0C,
    ord("("): 0x28,
    ord(")"): 0x29,
    ord("\\"): 0x5C,
}


class TokenKind(Enum):
    NUMBER = "number"
    NAME = "name"
    LITERAL_STRING = "literal_string"
    HEX_STRING = "hex_string"
    ARRAY_START = "array_start"
    ARRAY_END = "array_end"
    DICT_START = "dict_start"
    DICT_END = "dict_end"
    KEYWORD = "keyword"
    EOF = "eof"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Token:
    """One lexical token.

    `value` is interpreted by kind: `bytes` (raw, undecoded) for NUMBER, KEYWORD,
    LITERAL_STRING, HEX_STRING and UNKNOWN; `str` (already `#xx`-unescaped) for
    NAME; `None` for structural/EOF tokens. `start`/`end` are byte offsets into
    the tokenizer's `data`, with `end` exclusive — the natural next scan position.
    """

    kind: TokenKind
    value: bytes | str | None
    start: int
    end: int


class PdfTokenizer:
    """Tokenizes PDF COS syntax directly out of a byte string, at arbitrary offsets."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    @property
    def data(self) -> bytes:
        return self._data

    def next_token(self, offset: int) -> Token:
        """Return the next token at or after `offset`, skipping whitespace/comments."""
        data = self._data
        length = len(data)
        pos = offset
        while pos < length:
            byte = data[pos]
            if byte in _WHITESPACE:
                pos += 1
                continue
            if byte == 0x25:  # '%' comment to end of line
                while pos < length and data[pos] not in (0x0A, 0x0D):
                    pos += 1
                continue
            break

        if pos >= length:
            return Token(TokenKind.EOF, None, length, length)

        byte = data[pos]

        if byte == ord("/"):
            return self._read_name(pos)
        if byte == ord("("):
            return self._read_literal_string(pos)
        if byte == ord("<"):
            if pos + 1 < length and data[pos + 1] == ord("<"):
                return Token(TokenKind.DICT_START, None, pos, pos + 2)
            return self._read_hex_string(pos)
        if byte == ord(">"):
            if pos + 1 < length and data[pos + 1] == ord(">"):
                return Token(TokenKind.DICT_END, None, pos, pos + 2)
            return Token(TokenKind.UNKNOWN, data[pos : pos + 1], pos, pos + 1)
        if byte == ord("["):
            return Token(TokenKind.ARRAY_START, None, pos, pos + 1)
        if byte == ord("]"):
            return Token(TokenKind.ARRAY_END, None, pos, pos + 1)
        if byte in _NUMBER_LEAD:
            number_token = self._try_read_number(pos)
            if number_token is not None:
                return number_token

        return self._read_keyword_or_unknown(pos)

    def _read_name(self, start: int) -> Token:
        data = self._data
        length = len(data)
        pos = start + 1
        chars = bytearray()
        while pos < length and data[pos] not in _WHITESPACE and data[pos] not in _DELIMITERS:
            if (
                data[pos] == ord("#")
                and pos + 2 < length
                and data[pos + 1] in _HEX_DIGITS
                and data[pos + 2] in _HEX_DIGITS
            ):
                chars.append(int(data[pos + 1 : pos + 3], 16))
                pos += 3
            else:
                chars.append(data[pos])
                pos += 1
        return Token(TokenKind.NAME, chars.decode("latin-1"), start, pos)

    def _read_literal_string(self, start: int) -> Token:
        data = self._data
        length = len(data)
        pos = start + 1
        depth = 1
        result = bytearray()
        while pos < length and depth > 0:
            ch = data[pos]
            if ch == ord("\\"):
                pos += 1
                if pos >= length:
                    break
                esc = data[pos]
                if esc in _ESCAPE_MAP:
                    result.append(_ESCAPE_MAP[esc])
                    pos += 1
                elif esc in (0x0D, 0x0A):
                    pos += 1
                    if esc == 0x0D and pos < length and data[pos] == 0x0A:
                        pos += 1
                elif 0x30 <= esc <= 0x37:
                    digits = bytearray([esc])
                    pos += 1
                    for _ in range(2):
                        if pos < length and 0x30 <= data[pos] <= 0x37:
                            digits.append(data[pos])
                            pos += 1
                        else:
                            break
                    result.append(int(bytes(digits), 8) & 0xFF)
                else:
                    result.append(esc)
                    pos += 1
            elif ch == ord("("):
                depth += 1
                result.append(ch)
                pos += 1
            elif ch == ord(")"):
                depth -= 1
                pos += 1
                if depth > 0:
                    result.append(ch)
            else:
                result.append(ch)
                pos += 1
        return Token(TokenKind.LITERAL_STRING, bytes(result), start, pos)

    def _read_hex_string(self, start: int) -> Token:
        data = self._data
        length = len(data)
        pos = start + 1
        hex_chars = bytearray()
        while pos < length and data[pos] != ord(">"):
            if data[pos] in _HEX_DIGITS:
                hex_chars.append(data[pos])
            pos += 1
        if pos < length:
            pos += 1  # consume '>'
        if len(hex_chars) % 2 == 1:
            hex_chars.append(ord("0"))
        try:
            value = bytes.fromhex(hex_chars.decode("ascii"))
        except ValueError:
            value = b""
        return Token(TokenKind.HEX_STRING, value, start, pos)

    def _try_read_number(self, start: int) -> Token | None:
        data = self._data
        length = len(data)
        pos = start
        if data[pos] in (ord("+"), ord("-")):
            pos += 1
        digits_before = pos
        while pos < length and 0x30 <= data[pos] <= 0x39:
            pos += 1
        has_int_part = pos > digits_before
        has_dot = False
        if pos < length and data[pos] == ord("."):
            has_dot = True
            pos += 1
            while pos < length and 0x30 <= data[pos] <= 0x39:
                pos += 1
        if not has_int_part and not has_dot:
            return None
        return Token(TokenKind.NUMBER, data[start:pos], start, pos)

    def _read_keyword_or_unknown(self, start: int) -> Token:
        data = self._data
        length = len(data)
        pos = start
        while pos < length and data[pos] not in _WHITESPACE and data[pos] not in _DELIMITERS:
            pos += 1
        if pos == start:
            return Token(TokenKind.UNKNOWN, data[start : start + 1], start, start + 1)
        return Token(TokenKind.KEYWORD, data[start:pos], start, pos)
