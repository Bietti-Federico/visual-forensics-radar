from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction


def test_fields() -> None:
    prediction = ModelPrediction(model_id="random_forest", probability=0.8, predicted_label=True)
    assert prediction.model_id == "random_forest"
    assert prediction.probability == 0.8
    assert prediction.predicted_label is True
