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


def test_flags_anses_prediction_without_acroform() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": False})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, _report("ANSES", 0.9))

    assert finding is not None
    assert finding.rule_id == "entity_template_mismatch"
    assert "ANSES" in finding.explanation


def test_does_not_flag_anses_prediction_with_acroform() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": True})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, _report("ANSES", 0.9))

    assert finding is None


def test_flags_la_rioja_prediction_with_unexpected_acroform() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": True})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, _report("LA_RIOJA", 0.8))

    assert finding is not None
    assert "LA_RIOJA" in finding.explanation


def test_low_confidence_prediction_does_not_trigger() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": False})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, _report("ANSES", 0.5))

    assert finding is None


def test_unknown_entity_does_not_trigger() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": False})
    finding = EntityTemplateMismatchRule().evaluate(
        feature_set, _report("SYNTHETIC_REFERENCE", 0.99)
    )

    assert finding is None


def test_missing_feature_does_not_trigger() -> None:
    feature_set = build_feature_set({})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, _report("ANSES", 0.9))

    assert finding is None


def test_empty_entity_report_does_not_trigger() -> None:
    feature_set = build_feature_set({"catalog.has_acroform": False})
    finding = EntityTemplateMismatchRule().evaluate(feature_set, EntityIdentificationReport())

    assert finding is None
