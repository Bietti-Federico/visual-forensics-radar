from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport
from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction


def test_by_model_returns_matching_prediction_or_none() -> None:
    a = ModelPrediction(model_id="random_forest", probability=0.1, predicted_label=False)
    b = ModelPrediction(model_id="xgboost", probability=0.9, predicted_label=True)
    report = MlEnsembleReport(predictions=[a, b])

    assert report.by_model("random_forest") is a
    assert report.by_model("xgboost") is b
    assert report.by_model("missing") is None


def test_len_and_iter() -> None:
    predictions = [
        ModelPrediction(model_id="a", probability=0.1, predicted_label=False),
        ModelPrediction(model_id="b", probability=0.9, predicted_label=True),
    ]
    report = MlEnsembleReport(predictions=predictions)
    assert len(report) == 2
    assert list(report) == predictions


def test_empty_report_defaults() -> None:
    report = MlEnsembleReport()
    assert len(report) == 0
    assert report.by_model("anything") is None
