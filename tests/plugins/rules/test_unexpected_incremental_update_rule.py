from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.unexpected_incremental_update_rule import (
    UnexpectedIncrementalUpdateRule,
)
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_above_default_threshold() -> None:
    feature_set = build_feature_set({"incremental_updates.revision_count": 3})
    finding = UnexpectedIncrementalUpdateRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_silent_at_default_threshold() -> None:
    feature_set = build_feature_set({"incremental_updates.revision_count": 2})
    assert UnexpectedIncrementalUpdateRule().evaluate(feature_set) is None


def test_custom_threshold() -> None:
    feature_set = build_feature_set({"incremental_updates.revision_count": 5})
    assert UnexpectedIncrementalUpdateRule(threshold=10).evaluate(feature_set) is None
    assert UnexpectedIncrementalUpdateRule(threshold=4).evaluate(feature_set) is not None


def test_silent_when_feature_missing() -> None:
    assert UnexpectedIncrementalUpdateRule().evaluate(build_feature_set({})) is None
