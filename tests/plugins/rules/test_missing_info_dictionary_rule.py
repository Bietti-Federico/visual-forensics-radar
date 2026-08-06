from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.missing_info_dictionary_rule import MissingInfoDictionaryRule
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_when_info_dict_missing() -> None:
    feature_set = build_feature_set({"metadata.has_info_dict": False})
    finding = MissingInfoDictionaryRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_silent_when_info_dict_present() -> None:
    feature_set = build_feature_set({"metadata.has_info_dict": True})
    assert MissingInfoDictionaryRule().evaluate(feature_set) is None


def test_silent_when_feature_missing() -> None:
    assert MissingInfoDictionaryRule().evaluate(build_feature_set({})) is None
