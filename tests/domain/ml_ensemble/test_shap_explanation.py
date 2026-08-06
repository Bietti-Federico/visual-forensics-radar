from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation


def test_fields() -> None:
    explanation = ShapExplanation(
        model_id="random_forest",
        feature_names=("a", "b"),
        shap_values=(0.1, -0.2),
        base_value=0.5,
    )
    assert explanation.feature_names == ("a", "b")
    assert explanation.shap_values == (0.1, -0.2)
    assert explanation.base_value == 0.5
