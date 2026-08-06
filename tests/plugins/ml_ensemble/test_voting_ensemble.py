import pytest

from pdf_forensics.plugins.ml_ensemble.voting_ensemble import VotingEnsembleModel
from tests.fixtures.ml_ensemble_fixtures import (
    HELD_OUT_MANIPULATED,
    HELD_OUT_ORIGINAL,
    TRAINING_LABELS,
    TRAINING_VECTORS,
)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="voting_ensemble"):
        VotingEnsembleModel().predict(HELD_OUT_ORIGINAL)


def test_predicts_correct_label_for_each_cluster() -> None:
    model = VotingEnsembleModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)

    original_prediction = model.predict(HELD_OUT_ORIGINAL)
    manipulated_prediction = model.predict(HELD_OUT_MANIPULATED)

    assert original_prediction.predicted_label is False
    assert manipulated_prediction.predicted_label is True


def test_explain_raises_not_implemented() -> None:
    model = VotingEnsembleModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)
    with pytest.raises(NotImplementedError, match="voting_ensemble"):
        model.explain(HELD_OUT_MANIPULATED)


def test_model_id() -> None:
    assert VotingEnsembleModel().model_id == "voting_ensemble"
