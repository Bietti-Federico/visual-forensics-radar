from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.plugins.rules.pdf_version_mismatch_rule import PdfVersionMismatchRule
from tests.fixtures.feature_helpers import build_feature_set


def test_triggers_when_versions_differ() -> None:
    feature_set = build_feature_set({"general.pdf_version": "1.4", "catalog.version": "1.7"})
    finding = PdfVersionMismatchRule().evaluate(feature_set)
    assert finding is not None
    assert finding.severity is AnomalySeverity.INFO


def test_silent_when_versions_match() -> None:
    feature_set = build_feature_set({"general.pdf_version": "1.7", "catalog.version": "1.7"})
    assert PdfVersionMismatchRule().evaluate(feature_set) is None


def test_silent_when_catalog_version_absent() -> None:
    feature_set = build_feature_set({"general.pdf_version": "1.7", "catalog.version": None})
    assert PdfVersionMismatchRule().evaluate(feature_set) is None


def test_silent_when_features_missing() -> None:
    assert PdfVersionMismatchRule().evaluate(build_feature_set({})) is None
