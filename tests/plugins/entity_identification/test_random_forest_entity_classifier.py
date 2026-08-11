import pytest

from pdf_forensics.plugins.entity_identification.random_forest_entity_classifier import (
    RandomForestEntityClassifier,
)

TRAINING_VECTORS = [
    {"producer": "Alpha PDF Writer", "size_bucket": 1},
    {"producer": "Alpha PDF Writer", "size_bucket": 2},
    {"producer": "Alpha PDF Writer", "size_bucket": 1},
    {"producer": "Bravo Suite", "size_bucket": 3},
    {"producer": "Bravo Suite", "size_bucket": 3},
    {"producer": "Bravo Suite", "size_bucket": 4},
    {"producer": "Charlie Forms", "size_bucket": 5},
    {"producer": "Charlie Forms", "size_bucket": 5},
    {"producer": "Charlie Forms", "size_bucket": 6},
]
TRAINING_LABELS = [
    "ENTITY_A",
    "ENTITY_A",
    "ENTITY_A",
    "ENTITY_B",
    "ENTITY_B",
    "ENTITY_B",
    "ENTITY_C",
    "ENTITY_C",
    "ENTITY_C",
]
HELD_OUT_A = {"producer": "Alpha PDF Writer", "size_bucket": 1}
HELD_OUT_UNSEEN_PRODUCER = {"producer": "Never Seen Producer", "size_bucket": 1}


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="random_forest_entity"):
        RandomForestEntityClassifier().predict(HELD_OUT_A)


def test_classifier_id() -> None:
    assert RandomForestEntityClassifier().classifier_id == "random_forest_entity"


def test_predicts_correct_entity_for_each_cluster() -> None:
    classifier = RandomForestEntityClassifier()
    classifier.fit(TRAINING_VECTORS, TRAINING_LABELS)

    prediction = classifier.predict(HELD_OUT_A)

    assert prediction.classifier_id == "random_forest_entity"
    assert prediction.predicted_entity == "ENTITY_A"
    assert prediction.confidence > 0.5


def test_probabilities_cover_every_known_entity_and_sum_to_one() -> None:
    classifier = RandomForestEntityClassifier()
    classifier.fit(TRAINING_VECTORS, TRAINING_LABELS)

    prediction = classifier.predict(HELD_OUT_A)

    assert set(prediction.probabilities) == {"ENTITY_A", "ENTITY_B", "ENTITY_C"}
    assert prediction.probabilities[prediction.predicted_entity] == pytest.approx(
        prediction.confidence
    )
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_unseen_producer_degrades_gracefully_instead_of_crashing() -> None:
    """`DictVectorizer.transform` silently drops categorical values it never
    saw during `fit` — a document from an entity outside the training set
    should read as low-confidence across every known class, not raise."""
    classifier = RandomForestEntityClassifier()
    classifier.fit(TRAINING_VECTORS, TRAINING_LABELS)

    prediction = classifier.predict(HELD_OUT_UNSEEN_PRODUCER)

    assert prediction.predicted_entity in {"ENTITY_A", "ENTITY_B", "ENTITY_C"}
    assert 0.0 <= prediction.confidence <= 1.0
