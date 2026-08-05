"""Walks the `/Prev` chain of incremental updates back to the original revision.

Takes a `parse_revision_at` callback rather than doing any byte parsing itself —
this module's only job is the chain-walking *algorithm* (cycle detection, missing
`/Root` detection), independent of whether a given revision turns out to be a
classic xref table or an xref stream. That keeps it trivially unit-testable with
a fake in-memory `{offset: Revision}` lookup.
"""

from __future__ import annotations

from collections.abc import Callable

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector, AnomalySeverity
from pdf_forensics.domain.pdf.document import Revision


def walk_revision_chain(
    initial_offset: int,
    parse_revision_at: Callable[[int], Revision | None],
    anomalies: AnomalyCollector,
) -> list[Revision]:
    """Follow `/Prev` from the most recent revision backward to the original save.

    Returns revisions ordered most-recent-first. Stops (without raising) on a
    cycle, an offset that doesn't yield a parseable revision, or a missing
    `/Prev` — all are recorded as anomalies rather than aborting the parse.
    """
    revisions: list[Revision] = []
    seen_offsets: set[int] = set()
    offset: int | None = initial_offset

    while offset is not None:
        if offset in seen_offsets:
            anomalies.record(
                AnomalyCode.BROKEN_PREV_CHAIN,
                AnomalySeverity.WARNING,
                f"/Prev chain revisits byte offset {offset}; stopped to avoid an infinite loop.",
                byte_offset=offset,
            )
            break
        seen_offsets.add(offset)

        revision = parse_revision_at(offset)
        if revision is None:
            break
        revisions.append(revision)
        offset = revision.trailer.get_int("Prev")

    if revisions and not _has_root(revisions):
        anomalies.record(
            AnomalyCode.TRAILER_MISSING_ROOT,
            AnomalySeverity.WARNING,
            "No /Root key found in any revision's trailer.",
        )

    return revisions


def _has_root(revisions: list[Revision]) -> bool:
    return any(revision.trailer.get_ref("Root") is not None for revision in revisions)
