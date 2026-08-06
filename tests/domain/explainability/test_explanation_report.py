from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.explainability.top_feature_contribution import TopFeatureContribution


def test_defaults_are_empty() -> None:
    report = ExplanationReport()
    assert report.reasons == []
    assert report.top_features == []


def test_holds_reasons_and_top_features() -> None:
    contribution = TopFeatureContribution(model_id="xgboost", feature_name="a", shap_value=0.1)
    report = ExplanationReport(reasons=["reason 1"], top_features=[contribution])
    assert report.reasons == ["reason 1"]
    assert report.top_features == [contribution]
