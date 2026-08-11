import pytest

from pdf_forensics.plugins.ml_ensemble.random_forest_classifier import RandomForestModel


def test_fit_with_only_one_class_raises_a_clear_error() -> None:
    # predict_proba on a model fit with a single class returns a one-column
    # array; predict()'s `[0][1]` would previously raise an opaque IndexError.
    vectors = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    labels = [True, True, True]

    with pytest.raises(ValueError, match="both genuine and manipulated"):
        RandomForestModel().fit(vectors, labels)
