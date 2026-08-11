from pdf_forensics.application.entity_invariants.check_entity_invariants_use_case import (
    CheckEntityInvariantsUseCase,
)
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity


def _feature(name: str, value: object, value_type: FeatureType) -> Feature:
    return Feature(
        name=name,
        value=value,
        value_type=value_type,
        description="test feature",
        source="test",
        confidence=1.0,
        category=FeatureCategory.GENERAL,
        schema_version="1.0.0",
    )


def _feature_set(**typed_values: tuple[object, FeatureType]) -> FeatureSet:
    return FeatureSet(
        features=[
            _feature(name, value, value_type) for name, (value, value_type) in typed_values.items()
        ]
    )


def _report(entity: str, confidence: float) -> EntityIdentificationReport:
    return EntityIdentificationReport(
        predictions=[
            EntityPrediction(
                classifier_id="test_classifier",
                predicted_entity=entity,
                confidence=confidence,
                probabilities={entity: confidence},
            )
        ]
    )


def test_mismatched_value_produces_warning_finding_naming_the_violation() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set(**{"catalog.has_acroform": (False, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is not None
    assert finding.rule_id == "entity_template_mismatch"
    assert "formulario o firma digital" in finding.explanation
    assert "ANSES" in finding.explanation


def test_matching_value_produces_no_finding() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set(**{"catalog.has_acroform": (True, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is None


def test_missing_feature_on_checked_document_is_skipped_not_a_violation() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set()  # feature entirely absent

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is None


def test_low_confidence_prediction_suppresses_the_check() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set(**{"catalog.has_acroform": (False, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.5))

    assert finding is None


def test_no_invariants_produces_no_finding() -> None:
    feature_set = _feature_set(**{"catalog.has_acroform": (False, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase([]).execute(feature_set, _report("ANSES", 0.9))

    assert finding is None


def test_no_predictions_produces_no_finding() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set(**{"catalog.has_acroform": (False, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(
        feature_set, EntityIdentificationReport()
    )

    assert finding is None


def test_minority_of_violations_is_warning_severity() -> None:
    invariants = [
        LearnedInvariant("catalog.has_acroform", True),
        LearnedInvariant("metadata.has_creator", False),
        LearnedInvariant("metadata.has_mod_date", True),
    ]
    feature_set = _feature_set(
        **{
            "catalog.has_acroform": (False, FeatureType.BOOLEAN),  # violated
            "metadata.has_creator": (False, FeatureType.BOOLEAN),  # matches
            "metadata.has_mod_date": (True, FeatureType.BOOLEAN),  # matches
        }
    )

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is not None
    assert finding.severity is AnomalySeverity.WARNING


def test_majority_of_violations_is_critical_severity() -> None:
    invariants = [
        LearnedInvariant("catalog.has_acroform", True),
        LearnedInvariant("metadata.has_creator", False),
        LearnedInvariant("metadata.has_mod_date", True),
    ]
    feature_set = _feature_set(
        **{
            "catalog.has_acroform": (False, FeatureType.BOOLEAN),  # violated
            "metadata.has_creator": (True, FeatureType.BOOLEAN),  # violated
            "metadata.has_mod_date": (True, FeatureType.BOOLEAN),  # matches
        }
    )

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is not None
    assert finding.severity is AnomalySeverity.CRITICAL


def test_force_bypasses_the_confidence_gate() -> None:
    invariants = [LearnedInvariant("catalog.has_acroform", True)]
    feature_set = _feature_set(**{"catalog.has_acroform": (False, FeatureType.BOOLEAN)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(
        feature_set, _report("ANSES", 0.5), force=True
    )

    assert finding is not None


def test_histogram_derived_invariant_is_checked_against_dict_feature() -> None:
    invariants = [LearnedInvariant("streams.filter_histogram::DCTDecode", 0)]
    feature_set = _feature_set(**{"streams.filter_histogram": ({"DCTDecode": 1}, FeatureType.DICT)})

    finding = CheckEntityInvariantsUseCase(invariants).execute(feature_set, _report("ANSES", 0.9))

    assert finding is not None
    assert "cantidad de streams con filtro DCTDecode" in finding.explanation
