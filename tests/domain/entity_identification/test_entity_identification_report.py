from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction


def _prediction(classifier_id: str, entity: str, confidence: float) -> EntityPrediction:
    return EntityPrediction(
        classifier_id=classifier_id,
        predicted_entity=entity,
        confidence=confidence,
        probabilities={entity: confidence},
    )


def test_by_classifier_finds_matching_prediction() -> None:
    prediction = _prediction("random_forest_entity", "ANSES", 0.9)
    report = EntityIdentificationReport(predictions=[prediction])

    assert report.by_classifier("random_forest_entity") is prediction
    assert report.by_classifier("missing") is None


def test_len_and_iteration() -> None:
    predictions = [_prediction("a", "ANSES", 0.8), _prediction("b", "LA_RIOJA", 0.6)]
    report = EntityIdentificationReport(predictions=predictions)

    assert len(report) == 2
    assert list(report) == predictions


def test_empty_report_has_no_predictions() -> None:
    report = EntityIdentificationReport()

    assert len(report) == 0
    assert report.by_classifier("anything") is None
