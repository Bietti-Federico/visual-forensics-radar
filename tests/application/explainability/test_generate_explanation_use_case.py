from pdf_forensics.application.explainability.generate_explanation_use_case import (
    GenerateExplanationUseCase,
)
from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore
from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport
from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport


def _finding(rule_id: str, severity: AnomalySeverity) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id,
        severity=severity,
        confidence=1.0,
        explanation=f"explanation for {rule_id}",
        references=(),
    )


def test_reasons_ordered_critical_then_warning_then_info() -> None:
    rule_report = RuleEvaluationReport(
        findings=[
            _finding("info_rule", AnomalySeverity.INFO),
            _finding("critical_rule", AnomalySeverity.CRITICAL),
            _finding("warning_rule", AnomalySeverity.WARNING),
        ]
    )
    report = GenerateExplanationUseCase().execute(
        rule_report, AnomalyDetectionReport(), MlEnsembleReport(), []
    )

    assert report.reasons == [
        "explanation for critical_rule",
        "explanation for warning_rule",
        "explanation for info_rule",
    ]


def test_non_triggering_signals_produce_no_extra_reasons() -> None:
    anomaly_report = AnomalyDetectionReport(
        scores=[AnomalyScore(detector_id="isolation_forest", score=0.1, is_anomaly=False)]
    )
    ml_report = MlEnsembleReport(
        predictions=[ModelPrediction(model_id="xgboost", probability=0.2, predicted_label=False)]
    )
    report = GenerateExplanationUseCase().execute(
        RuleEvaluationReport(), anomaly_report, ml_report, []
    )
    assert report.reasons == []


def test_anomalous_score_and_manipulated_prediction_add_reasons() -> None:
    anomaly_report = AnomalyDetectionReport(
        scores=[AnomalyScore(detector_id="isolation_forest", score=1.234, is_anomaly=True)]
    )
    ml_report = MlEnsembleReport(
        predictions=[ModelPrediction(model_id="xgboost", probability=0.87, predicted_label=True)]
    )
    report = GenerateExplanationUseCase().execute(
        RuleEvaluationReport(), anomaly_report, ml_report, []
    )

    assert report.reasons == [
        "el detector de aislamiento (Isolation Forest) marcó este documento como fuera de "
        "lo normal para esta entidad (score interno=1.234, no comparable entre detectores).",
        "el modelo XGBoost predice que este documento está manipulado (probabilidad=87%).",
    ]


def test_top_features_truncated_and_tagged_per_model() -> None:
    explanation = ShapExplanation(
        model_id="random_forest",
        feature_names=("a", "b", "c", "d"),
        shap_values=(0.1, -0.9, 0.5, -0.05),
        base_value=0.3,
    )
    report = GenerateExplanationUseCase(top_n_features=2).execute(
        RuleEvaluationReport(), AnomalyDetectionReport(), MlEnsembleReport(), [explanation]
    )

    assert [f.feature_name for f in report.top_features] == ["b", "c"]
    assert all(f.model_id == "random_forest" for f in report.top_features)


def test_top_features_kept_separate_per_model() -> None:
    explanation_a = ShapExplanation(
        model_id="random_forest",
        feature_names=("a",),
        shap_values=(1.0,),
        base_value=0.0,
    )
    explanation_b = ShapExplanation(
        model_id="logistic_regression",
        feature_names=("a",),
        shap_values=(0.5,),
        base_value=0.0,
    )
    report = GenerateExplanationUseCase().execute(
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [explanation_a, explanation_b],
    )

    model_ids = [f.model_id for f in report.top_features]
    assert model_ids == ["random_forest", "logistic_regression"]


def test_empty_everything_returns_empty_report() -> None:
    report = GenerateExplanationUseCase().execute(
        RuleEvaluationReport(), AnomalyDetectionReport(), MlEnsembleReport(), []
    )
    assert report.reasons == []
    assert report.top_features == []
