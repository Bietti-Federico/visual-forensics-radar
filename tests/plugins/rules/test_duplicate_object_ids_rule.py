from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.duplicate_object_ids_rule import DuplicateObjectIdsRule
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_when_duplicates_present() -> None:
    feature_set = build_feature_set({"objects.duplicate_object_id_count": 2})
    finding = DuplicateObjectIdsRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING
    assert "2" in finding.explanation


def test_singular_wording_for_a_single_duplicate() -> None:
    feature_set = build_feature_set({"objects.duplicate_object_id_count": 1})
    finding = DuplicateObjectIdsRule().evaluate(feature_set)
    assert finding is not None
    assert "está definido" in finding.explanation
    assert "número de objeto" in finding.explanation


def test_silent_when_zero() -> None:
    feature_set = build_feature_set({"objects.duplicate_object_id_count": 0})
    assert DuplicateObjectIdsRule().evaluate(feature_set) is None


def test_silent_when_feature_missing() -> None:
    assert DuplicateObjectIdsRule().evaluate(build_feature_set({})) is None
