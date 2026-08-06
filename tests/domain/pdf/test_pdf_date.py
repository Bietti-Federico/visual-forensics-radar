from datetime import UTC

from pdf_forensics.domain.pdf.pdf_date import parse_pdf_date


def test_parses_date_with_explicit_offset() -> None:
    parsed = parse_pdf_date("D:20260722173545-03'00'")
    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert parsed.isoformat() == "2026-07-22T20:35:45+00:00"


def test_parses_date_with_z_timezone() -> None:
    parsed = parse_pdf_date("D:20260722173545Z")
    assert parsed is not None
    assert parsed.isoformat() == "2026-07-22T17:35:45+00:00"


def test_two_equivalent_instants_in_different_timezones_compare_equal() -> None:
    a = parse_pdf_date("D:20260722173545-03'00'")
    b = parse_pdf_date("D:20260722203545Z")
    assert a == b


def test_none_when_raw_is_none_or_empty() -> None:
    assert parse_pdf_date(None) is None
    assert parse_pdf_date("") is None


def test_none_when_no_timezone() -> None:
    assert parse_pdf_date("D:20260722173545") is None


def test_none_when_no_hour_component() -> None:
    assert parse_pdf_date("D:20260722") is None


def test_none_when_malformed() -> None:
    assert parse_pdf_date("not a pdf date") is None


def test_none_when_invalid_calendar_date() -> None:
    assert parse_pdf_date("D:20260231120000Z") is None  # February 31st doesn't exist


def test_none_when_timezone_offset_unparseable() -> None:
    assert parse_pdf_date("D:20260722173545+garbage") is None
