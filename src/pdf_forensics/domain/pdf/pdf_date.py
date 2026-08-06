"""Parses PDF date strings (ISO 32000-1 §7.9.4) to timezone-aware UTC `datetime`s.

PDF date strings look like `D:20260722173545+00'00'` — ISO-ish but not ISO 8601.
Creation and modification dates are commonly stamped in *different* timezone
notations (e.g. local `-03'00'` vs a signing step that stamps `Z`/UTC) —
comparing the raw hour:minute:second fields directly, as a human eyeballing
two raw strings would, can make a same-instant save look like a multi-hour
gap or vice versa. Normalizing both to UTC first makes any comparison real.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

_PDF_DATE_PATTERN = re.compile(
    r"^D:(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
    r"(?P<hour>\d{2})?(?P<minute>\d{2})?(?P<second>\d{2})?"
    r"(?P<tz>[Z+\-].*)?$"
)
_TZ_OFFSET_PATTERN = re.compile(r"^([+\-])(\d{2})'?(\d{2})'?$")


def parse_pdf_date(raw: str | None) -> datetime | None:
    """Parse a PDF date string to a UTC `datetime`, or `None` if that's not possible.

    Returns `None` when `raw` is missing, doesn't match the PDF date grammar,
    has no hour component, or has no (or an unparseable) timezone offset —
    a bare local time with no way to normalize it to UTC isn't comparable to
    another date, so it's better to report "unknown" than guess.
    """
    if not raw:
        return None
    match = _PDF_DATE_PATTERN.match(raw)
    if not match:
        return None
    groups = match.groupdict()
    if groups["hour"] is None:
        return None
    offset = _parse_tz_offset(groups["tz"])
    if offset is None:
        return None
    try:
        naive = datetime(
            int(groups["year"]),
            int(groups["month"]),
            int(groups["day"]),
            int(groups["hour"]),
            int(groups["minute"] or 0),
            int(groups["second"] or 0),
        )
    except ValueError:
        return None
    return naive.replace(tzinfo=timezone(offset)).astimezone(UTC)


def _parse_tz_offset(tz_raw: str | None) -> timedelta | None:
    if not tz_raw:
        return None
    if tz_raw == "Z":
        return timedelta(0)
    match = _TZ_OFFSET_PATTERN.match(tz_raw)
    if not match:
        return None
    sign, hours, minutes = match.groups()
    delta = timedelta(hours=int(hours), minutes=int(minutes))
    return -delta if sign == "-" else delta
