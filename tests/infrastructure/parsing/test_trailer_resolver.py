from decimal import Decimal

from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalyCollector
from pdf_forensics.domain.pdf.document import Revision
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfNumber, PdfReference
from pdf_forensics.infrastructure.parsing.trailer_resolver import walk_revision_chain


def _revision(entries: dict, offset: int) -> Revision:
    return Revision(
        trailer=PdfDictionary(entries), xref_entries={}, xref_offset=offset, is_xref_stream=False
    )


def test_walks_prev_chain_in_order() -> None:
    revisions_by_offset = {
        100: _revision({"Prev": PdfNumber(Decimal(50), True)}, 100),
        50: _revision({"Root": PdfReference(1, 0)}, 50),
    }
    anomalies = AnomalyCollector()
    result = walk_revision_chain(100, revisions_by_offset.get, anomalies)
    assert [r.xref_offset for r in result] == [100, 50]
    assert not anomalies.to_list()


def test_stops_and_flags_cycle() -> None:
    revisions_by_offset = {
        100: _revision({"Prev": PdfNumber(Decimal(50), True)}, 100),
        50: _revision({"Prev": PdfNumber(Decimal(100), True), "Root": PdfReference(1, 0)}, 50),
    }
    anomalies = AnomalyCollector()
    result = walk_revision_chain(100, revisions_by_offset.get, anomalies)
    assert [r.xref_offset for r in result] == [100, 50]
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.BROKEN_PREV_CHAIN in codes


def test_flags_missing_root() -> None:
    revisions_by_offset = {100: _revision({}, 100)}
    anomalies = AnomalyCollector()
    walk_revision_chain(100, revisions_by_offset.get, anomalies)
    codes = [a.code for a in anomalies.to_list()]
    assert AnomalyCode.TRAILER_MISSING_ROOT in codes


def test_unparseable_offset_stops_without_raising() -> None:
    anomalies = AnomalyCollector()
    result = walk_revision_chain(999, lambda offset: None, anomalies)
    assert result == []
