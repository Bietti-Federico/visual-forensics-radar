import pytest

from pdf_forensics.plugins.ml_ensemble.random_forest_classifier import RandomForestModel
from tests.fixtures.ml_ensemble_fixtures import (
    HELD_OUT_MANIPULATED,
    HELD_OUT_ORIGINAL,
    TRAINING_LABELS,
    TRAINING_VECTORS,
)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="random_forest"):
        RandomForestModel().predict(HELD_OUT_ORIGINAL)


def test_explain_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="random_forest"):
        RandomForestModel().explain(HELD_OUT_ORIGINAL)


def test_predicts_correct_label_for_each_cluster() -> None:
    model = RandomForestModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)

    original_prediction = model.predict(HELD_OUT_ORIGINAL)
    manipulated_prediction = model.predict(HELD_OUT_MANIPULATED)

    assert original_prediction.predicted_label is False
    assert manipulated_prediction.predicted_label is True
    assert manipulated_prediction.probability > original_prediction.probability


def test_explain_returns_one_shap_value_per_feature() -> None:
    model = RandomForestModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)

    explanation = model.explain(HELD_OUT_MANIPULATED)

    assert explanation.model_id == "random_forest"
    assert len(explanation.feature_names) == 2
    assert len(explanation.shap_values) == 2
    assert all(isinstance(v, float) for v in explanation.shap_values)


def test_model_id() -> None:
    assert RandomForestModel().model_id == "random_forest"
