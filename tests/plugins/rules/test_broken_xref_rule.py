from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.broken_xref_rule import BrokenXrefRule
from tests.fixtures.feature_helpers import build_feature_set


def test_critical_when_unparseable_fallback_used() -> None:
    feature_set = build_feature_set(
        {"statistics.anomaly_code_histogram": {"xref_unparseable_fallback_used": 1}}
    )
    finding = BrokenXrefRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.CRITICAL


def test_warning_when_offset_mismatch_present() -> None:
    feature_set = build_feature_set(
        {"statistics.anomaly_code_histogram": {"xref_offset_mismatch": 1}}
    )
    finding = BrokenXrefRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_warning_when_broken_prev_chain_present() -> None:
    feature_set = build_feature_set({"statistics.anomaly_code_histogram": {"broken_prev_chain": 1}})
    finding = BrokenXrefRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_silent_when_no_xref_anomalies() -> None:
    feature_set = build_feature_set(
        {"statistics.anomaly_code_histogram": {"duplicate_object_id": 1}}
    )
    assert BrokenXrefRule().evaluate(feature_set) is None


def test_silent_when_histogram_empty() -> None:
    feature_set = build_feature_set({"statistics.anomaly_code_histogram": {}})
    assert BrokenXrefRule().evaluate(feature_set) is None


def test_silent_when_feature_missing() -> None:
    assert BrokenXrefRule().evaluate(build_feature_set({})) is None
