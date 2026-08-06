from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction
from pdf_forensics.plugins.rules.entity_template_mismatch_rule import EntityTemplateMismatchRule
from tests.fixtures.feature_helpers import build_feature_set


def _report(entity: str, confidence: float) -> EntityIdentificationReport:
    return EntityIdentificationReport(
        predictions=[
            EntityPrediction(
                classifier_id="random_forest_entity",
                predicted_entity=entity,
                confidence=confidence,
                probabilities={entity: confidence},
            )
        ]
    )


def _anses_conforming_features() -> dict:
    return {
        "catalog.has_acroform": True,
        "streams.filter_histogram": {"FlateDecode": 14},
    }


def test_flags_anses_prediction_without_acroform() -> None:
    features = _anses_conforming_features()
    features["catalog.has_acroform"] = False
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("ANSES", 0.9)
    )

    assert finding is not None
    assert finding.rule_id == "entity_template_mismatch"
    assert "ANSES" in finding.explanation
    assert "AcroForm" in finding.explanation


def test_flags_anses_prediction_with_embedded_jpeg() -> None:
    features = _anses_conforming_features()
    features["streams.filter_histogram"] = {"FlateDecode": 12, "DCTDecode": 1}
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("ANSES", 0.9)
    )

    assert finding is not None
    assert "DCTDecode" in finding.explanation


def test_does_not_flag_fully_conforming_anses_prediction() -> None:
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(_anses_conforming_features()), _report("ANSES", 0.9)
    )
    assert finding is None


def test_flags_la_rioja_prediction_with_unexpected_acroform() -> None:
    features = {"catalog.has_acroform": True, "streams.filter_histogram": {"DCTDecode": 1}}
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("LA_RIOJA", 0.8)
    )

    assert finding is not None
    assert "LA_RIOJA" in finding.explanation


def test_flags_jujuy_prediction_with_wrong_image_count() -> None:
    features = {"catalog.has_acroform": False, "streams.filter_histogram": {"DCTDecode": 1}}
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("JUJUY", 0.8)
    )

    assert finding is not None
    assert "JUJUY" in finding.explanation


def test_low_confidence_prediction_does_not_trigger() -> None:
    features = _anses_conforming_features()
    features["catalog.has_acroform"] = False
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("ANSES", 0.5)
    )

    assert finding is None


def test_unknown_entity_does_not_trigger() -> None:
    features = {"catalog.has_acroform": False}
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), _report("SYNTHETIC_REFERENCE", 0.99)
    )

    assert finding is None


def test_missing_features_do_not_trigger() -> None:
    finding = EntityTemplateMismatchRule().evaluate(build_feature_set({}), _report("ANSES", 0.9))

    assert finding is None


def test_empty_entity_report_does_not_trigger() -> None:
    features = {"catalog.has_acroform": False}
    finding = EntityTemplateMismatchRule().evaluate(
        build_feature_set(features), EntityIdentificationReport()
    )

    assert finding is None
