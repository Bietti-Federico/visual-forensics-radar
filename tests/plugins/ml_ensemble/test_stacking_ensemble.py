import pytest

from pdf_forensics.plugins.ml_ensemble.stacking_ensemble import StackingEnsembleModel
from tests.fixtures.ml_ensemble_fixtures import (
    HELD_OUT_MANIPULATED,
    HELD_OUT_ORIGINAL,
    TRAINING_LABELS,
    TRAINING_VECTORS,
)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="stacking_ensemble"):
        StackingEnsembleModel().predict(HELD_OUT_ORIGINAL)


def test_predicts_correct_label_for_each_cluster() -> None:
    model = StackingEnsembleModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)

    original_prediction = model.predict(HELD_OUT_ORIGINAL)
    manipulated_prediction = model.predict(HELD_OUT_MANIPULATED)

    assert original_prediction.predicted_label is False
    assert manipulated_prediction.predicted_label is True


def test_explain_raises_not_implemented() -> None:
    model = StackingEnsembleModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)
    with pytest.raises(NotImplementedError, match="stacking_ensemble"):
        model.explain(HELD_OUT_MANIPULATED)


def test_model_id() -> None:
    assert StackingEnsembleModel().model_id == "stacking_ensemble"


def test_fits_at_the_documented_minimum_training_size() -> None:
    """5 genuine / 3 fraud is this platform's documented minimum training
    size. sklearn's default cv=5 for StackingClassifier crashes here since
    the minority class (3 fraud) can't support a 5-fold stratified split."""
    vectors = [{"x": float(i)} for i in range(8)]
    labels = [False] * 5 + [True] * 3

    model = StackingEnsembleModel()
    model.fit(vectors, labels)

    prediction = model.predict({"x": 100.0})
    assert prediction.predicted_label in (True, False)
