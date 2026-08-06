import pytest

from pdf_forensics.plugins.ml_ensemble.extra_trees_classifier import ExtraTreesModel
from tests.fixtures.ml_ensemble_fixtures import (
    HELD_OUT_MANIPULATED,
    HELD_OUT_ORIGINAL,
    TRAINING_LABELS,
    TRAINING_VECTORS,
)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="extra_trees"):
        ExtraTreesModel().predict(HELD_OUT_ORIGINAL)


def test_predicts_correct_label_for_each_cluster() -> None:
    model = ExtraTreesModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)

    original_prediction = model.predict(HELD_OUT_ORIGINAL)
    manipulated_prediction = model.predict(HELD_OUT_MANIPULATED)

    assert original_prediction.predicted_label is False
    assert manipulated_prediction.predicted_label is True


def test_explain_returns_one_shap_value_per_feature() -> None:
    model = ExtraTreesModel()
    model.fit(TRAINING_VECTORS, TRAINING_LABELS)
    explanation = model.explain(HELD_OUT_MANIPULATED)
    assert len(explanation.feature_names) == 2
    assert len(explanation.shap_values) == 2


def test_model_id() -> None:
    assert ExtraTreesModel().model_id == "extra_trees"
