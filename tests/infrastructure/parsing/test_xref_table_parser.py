from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector
from pdf_forensics.domain.pdf.document import XrefEntryType
from pdf_forensics.infrastructure.parsing.object_parser import ObjectParser
from pdf_forensics.infrastructure.parsing.tokenizer import PdfTokenizer
from pdf_forensics.infrastructure.parsing.xref_table_parser import parse_xref_table


def test_parses_classic_table_and_trailer() -> None:
    data = (
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"0000000020 00000 n \n"
        b"trailer\n"
        b"<< /Size 3 /Root 1 0 R >>\n"
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    revision = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)

    assert revision is not None
    assert revision.xref_entries[0].entry_type == XrefEntryType.FREE
    assert revision.xref_entries[1].entry_type == XrefEntryType.IN_USE
    assert revision.xref_entries[1].offset_or_stream_obj_num == 10
    assert revision.xref_entries[2].offset_or_stream_obj_num == 20
    assert revision.trailer.get_int("Size") == 3
    assert not anomalies.to_list()


def test_returns_none_when_not_an_xref_keyword() -> None:
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(b"not xref at all")
    result = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)
    assert result is None


def test_duplicate_entry_in_same_table_is_flagged_and_latest_wins() -> None:
    data = (
        b"xref\n"
        b"0 2\n"
        b"0000000000 65535 f \n"
        b"0000000010 00000 n \n"
        b"1 1\n"
        b"0000000099 00000 n \n"
        b"trailer\n"
        b"<< /Size 2 /Root 1 0 R >>\n"
    )
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    revision = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)

    assert revision is not None
    assert revision.xref_entries[1].offset_or_stream_obj_num == 99
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.DUPLICATE_OBJECT_ID in codes


def test_subsection_missing_count_stops_cleanly() -> None:
    # "0" alone with no count number: the subsection loop bails out immediately,
    # and since the parser can't tell how many stray tokens to skip, the trailer
    # search from that same position doesn't find "trailer" either — an empty,
    # tolerant result rather than a crash.
    data = b"xref\n0\ntrailer\n<< /Size 0 /Root 1 0 R >>\n"
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    revision = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)
    assert revision is not None
    assert revision.xref_entries == {}
    assert revision.trailer.get_int("Size") is None


def test_subsection_truncated_mid_entry_stops_cleanly() -> None:
    data = b"xref\n0 2\n0000000000 65535 f \ntrailer\n<< /Size 2 /Root 1 0 R >>\n"
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    revision = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)
    assert revision is not None
    assert revision.xref_entries[0].entry_type == XrefEntryType.FREE
    assert 1 not in revision.xref_entries


def test_missing_trailer_keyword_yields_empty_trailer() -> None:
    data = b"xref\n0 1\n0000000000 65535 f \n"
    anomalies = AnomalyCollector()
    tokenizer = PdfTokenizer(data)
    revision = parse_xref_table(tokenizer, ObjectParser(tokenizer, anomalies), 0, anomalies)
    assert revision is not None
    assert revision.trailer.get("Size") is None
