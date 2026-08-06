from pdf_forensics.domain.explainability.top_feature_contribution import TopFeatureContribution


def test_fields() -> None:
    contribution = TopFeatureContribution(
        model_id="xgboost", feature_name="general.object_count", shap_value=0.42
    )
    assert contribution.model_id == "xgboost"
    assert contribution.feature_name == "general.object_count"
    assert contribution.shap_value == 0.42
