from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.creation_after_mod_date_rule import CreationAfterModDateRule
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_when_creation_is_after_mod() -> None:
    feature_set = build_feature_set(
        {
            "metadata.creation_date": "D:20260722173545Z",
            "metadata.mod_date": "D:20260721173545Z",
        }
    )
    finding = CreationAfterModDateRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING
    assert finding.rule_id == "creation_after_mod_date"


def test_silent_when_creation_before_mod() -> None:
    feature_set = build_feature_set(
        {
            "metadata.creation_date": "D:20260721173545Z",
            "metadata.mod_date": "D:20260722173545Z",
        }
    )
    assert CreationAfterModDateRule().evaluate(feature_set) is None


def test_silent_when_dates_missing() -> None:
    assert CreationAfterModDateRule().evaluate(build_feature_set({})) is None


def test_silent_when_dates_unparseable() -> None:
    feature_set = build_feature_set(
        {"metadata.creation_date": "garbage", "metadata.mod_date": "D:20260722173545Z"}
    )
    assert CreationAfterModDateRule().evaluate(feature_set) is None


def test_silent_when_date_value_is_none() -> None:
    feature_set = build_feature_set({"metadata.creation_date": None, "metadata.mod_date": None})
    assert CreationAfterModDateRule().evaluate(feature_set) is None
