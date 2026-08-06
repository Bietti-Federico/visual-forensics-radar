from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.trailer_missing_root_rule import TrailerMissingRootRule
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_when_root_missing() -> None:
    feature_set = build_feature_set({"trailer.has_root": False})
    finding = TrailerMissingRootRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_silent_when_root_present() -> None:
    feature_set = build_feature_set({"trailer.has_root": True})
    assert TrailerMissingRootRule().evaluate(feature_set) is None


def test_silent_when_feature_missing() -> None:
    assert TrailerMissingRootRule().evaluate(build_feature_set({})) is None
