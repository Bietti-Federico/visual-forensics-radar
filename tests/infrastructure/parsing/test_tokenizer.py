from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer, Token, TokenKind


def _tokens(data: bytes) -> list[Token]:
    tokenizer = PdfTokenizer(data)
    tokens: list[Token] = []
    pos = 0
    while True:
        token = tokenizer.next_token(pos)
        tokens.append(token)
        if token.kind == TokenKind.EOF:
            break
        pos = token.end
    return tokens


def test_skips_whitespace_and_comments() -> None:
    tokens = _tokens(b"  % a comment\n 123")
    assert tokens[0].kind == TokenKind.NUMBER
    assert tokens[0].value == b"123"


def test_number_variants() -> None:
    for raw in (b"12", b"-3", b"+7", b"4.", b".5", b"-3.62"):
        tokenizer = PdfTokenizer(raw)
        token = tokenizer.next_token(0)
        assert token.kind == TokenKind.NUMBER
        assert token.value == raw
        assert token.end == len(raw)


def test_name_with_hash_escape() -> None:
    tokenizer = PdfTokenizer(b"/A#20B")
    token = tokenizer.next_token(0)
    assert token.kind == TokenKind.NAME
    assert token.value == "A B"


def test_name_stops_at_delimiter() -> None:
    tokenizer = PdfTokenizer(b"/Foo]")
    token = tokenizer.next_token(0)
    assert token.value == "Foo"
    assert token.end == 4


def test_literal_string_with_nested_parens_and_escapes() -> None:
    tokenizer = PdfTokenizer(rb"(a\(b\)c(nested)\n\061)")
    token = tokenizer.next_token(0)
    assert token.kind == TokenKind.LITERAL_STRING
    assert token.value == b"a(b)c(nested)\n1"


def test_literal_string_line_continuation_escape_is_dropped() -> None:
    tokenizer = PdfTokenizer(b"(line1\\\nline2)")
    token = tokenizer.next_token(0)
    assert token.value == b"line1line2"


def test_hex_string_odd_length_padded_with_zero() -> None:
    tokenizer = PdfTokenizer(b"<48656C6C6F1>")
    token = tokenizer.next_token(0)
    assert token.kind == TokenKind.HEX_STRING
    assert token.value == b"Hello\x10"


def test_hex_string_ignores_internal_whitespace() -> None:
    tokenizer = PdfTokenizer(b"<48 65 6C 6C 6F>")
    token = tokenizer.next_token(0)
    assert token.value == b"Hello"


def test_dict_start_vs_hex_string_disambiguation() -> None:
    tokenizer = PdfTokenizer(b"<< /A <FF> >>")
    assert tokenizer.next_token(0).kind == TokenKind.DICT_START
    name_tok = tokenizer.next_token(2)
    assert name_tok.kind == TokenKind.NAME
    hex_tok = tokenizer.next_token(name_tok.end)
    assert hex_tok.kind == TokenKind.HEX_STRING
    assert hex_tok.value == b"\xff"


def test_array_delimiters() -> None:
    tokens = _tokens(b"[1 2]")
    kinds = [t.kind for t in tokens]
    assert kinds == [
        TokenKind.ARRAY_START,
        TokenKind.NUMBER,
        TokenKind.NUMBER,
        TokenKind.ARRAY_END,
        TokenKind.EOF,
    ]


def test_keywords() -> None:
    for raw in (b"obj", b"endobj", b"stream", b"endstream", b"R", b"true", b"false", b"null"):
        tokenizer = PdfTokenizer(raw)
        token = tokenizer.next_token(0)
        assert token.kind == TokenKind.KEYWORD
        assert token.value == raw


def test_eof_at_end_of_data() -> None:
    tokenizer = PdfTokenizer(b"   ")
    token = tokenizer.next_token(0)
    assert token.kind == TokenKind.EOF


def test_lone_angle_bracket_is_unknown_not_a_crash() -> None:
    tokenizer = PdfTokenizer(b">")
    token = tokenizer.next_token(0)
    assert token.kind == TokenKind.UNKNOWN
    assert token.end == 1
