import pytest

from pdf_forensics.application.explainability.generate_explanation_use_case import (
    GenerateExplanationUseCase,
)
from pdf_forensics.application.risk_report.generate_risk_report_use_case import (
    GenerateRiskReportUseCase,
)
from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.entity_identification.entity_prediction import EntityPrediction
from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport
from pdf_forensics.domain.ml_ensemble.model_prediction import ModelPrediction
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.risk.risk_weights import RiskWeights
from pdf_forensics.domain.rules.rule_finding import RuleFinding
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport
from tests.fixtures.feature_helpers import build_feature_set


def _finding(severity: AnomalySeverity) -> RuleFinding:
    return RuleFinding(
        rule_id="test_rule", severity=severity, confidence=1.0, explanation="e", references=()
    )


def _clean_feature_set():
    return build_feature_set(
        {
            "statistics.anomaly_count_total": 0,
            "general.object_count": 5,
            "metadata.has_info_dict": True,
            "metadata.has_title": True,
            "metadata.has_author": True,
            "metadata.has_creator": True,
            "metadata.has_producer": True,
            "metadata.has_creation_date": True,
            "metadata.has_mod_date": True,
        }
    )


def test_all_zero_when_nothing_triggers() -> None:
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    assert report.risk_score == 0
    assert {c.name: c.score for c in report.components} == {
        "rule_engine": 0.0,
        "ml_probability": 0.0,
        "anomaly_detection": 0.0,
        "structural": 0.0,
        "metadata": 0.0,
        "entity_consistency": 0.0,
    }


def test_critical_findings_drive_rule_engine_component() -> None:
    rule_report = RuleEvaluationReport(findings=[_finding(AnomalySeverity.CRITICAL)])
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        rule_report,
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    rule_engine_component = next(c for c in report.components if c.name == "rule_engine")
    assert rule_engine_component.score == 0.9
    assert report.risk_score == round(0.9 * 0.27 * 100)


def test_noisy_or_saturates_but_never_exceeds_one() -> None:
    rule_report = RuleEvaluationReport(
        findings=[_finding(AnomalySeverity.CRITICAL) for _ in range(5)]
    )
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        rule_report,
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    rule_engine_component = next(c for c in report.components if c.name == "rule_engine")
    assert 0.999 < rule_engine_component.score < 1.0


def test_ml_probability_is_mean_of_predictions() -> None:
    ml_report = MlEnsembleReport(
        predictions=[
            ModelPrediction(model_id="a", probability=0.2, predicted_label=False),
            ModelPrediction(model_id="b", probability=0.8, predicted_label=True),
        ]
    )
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        ml_report,
        [],
        EntityIdentificationReport(),
    )
    ml_component = next(c for c in report.components if c.name == "ml_probability")
    assert ml_component.score == 0.5


def test_anomaly_detection_uses_is_anomaly_flag_only() -> None:
    anomaly_report = AnomalyDetectionReport(
        scores=[
            AnomalyScore(detector_id="a", score=999.0, is_anomaly=False),
            AnomalyScore(detector_id="b", score=0.01, is_anomaly=True),
        ]
    )
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        anomaly_report,
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    anomaly_component = next(c for c in report.components if c.name == "anomaly_detection")
    assert anomaly_component.score == 0.7


def test_structural_score_zero_when_features_absent() -> None:
    report = GenerateRiskReportUseCase().execute(
        build_feature_set({}),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    structural_component = next(c for c in report.components if c.name == "structural")
    assert structural_component.score == 0.0


def test_structural_score_from_anomaly_density() -> None:
    feature_set = build_feature_set(
        {"statistics.anomaly_count_total": 2, "general.object_count": 10}
    )
    report = GenerateRiskReportUseCase().execute(
        feature_set,
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    structural_component = next(c for c in report.components if c.name == "structural")
    assert structural_component.score == pytest.approx(min(1.0, 2 / 10 * 10))


def test_metadata_score_is_one_when_no_info_dict() -> None:
    feature_set = build_feature_set({"metadata.has_info_dict": False})
    report = GenerateRiskReportUseCase().execute(
        feature_set,
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    metadata_component = next(c for c in report.components if c.name == "metadata")
    assert metadata_component.score == 1.0


def test_metadata_score_is_fraction_missing() -> None:
    feature_set = build_feature_set(
        {
            "metadata.has_info_dict": True,
            "metadata.has_title": False,
            "metadata.has_author": False,
            "metadata.has_creator": True,
            "metadata.has_producer": True,
            "metadata.has_creation_date": True,
            "metadata.has_mod_date": True,
        }
    )
    report = GenerateRiskReportUseCase().execute(
        feature_set,
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    metadata_component = next(c for c in report.components if c.name == "metadata")
    assert metadata_component.score == 2 / 6


def test_custom_weights_change_final_score() -> None:
    rule_report = RuleEvaluationReport(findings=[_finding(AnomalySeverity.CRITICAL)])
    default_report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        rule_report,
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    custom_weights = RiskWeights(
        rule_engine=0.8,
        ml_probability=0.04,
        anomaly_detection=0.04,
        structural=0.04,
        metadata=0.04,
        entity_consistency=0.04,
    )
    custom_report = GenerateRiskReportUseCase(weights=custom_weights).execute(
        _clean_feature_set(),
        rule_report,
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
    )
    assert custom_report.risk_score > default_report.risk_score


def test_entity_consistency_score_is_one_minus_mean_confidence() -> None:
    entity_report = EntityIdentificationReport(
        predictions=[
            EntityPrediction(
                classifier_id="random_forest_entity",
                predicted_entity="ANSES",
                confidence=0.7,
                probabilities={"ANSES": 0.7, "LA_RIOJA": 0.2, "JUJUY": 0.1},
            )
        ]
    )
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        entity_report,
    )
    entity_component = next(c for c in report.components if c.name == "entity_consistency")
    assert entity_component.score == pytest.approx(0.3)


def test_embedded_explanation_matches_standalone_generation() -> None:
    rule_report = RuleEvaluationReport(findings=[_finding(AnomalySeverity.WARNING)])
    anomaly_report = AnomalyDetectionReport(
        scores=[AnomalyScore(detector_id="isolation_forest", score=1.0, is_anomaly=True)]
    )
    ml_report = MlEnsembleReport(
        predictions=[ModelPrediction(model_id="xgboost", probability=0.9, predicted_label=True)]
    )
    shap_explanations = [
        ShapExplanation(
            model_id="xgboost", feature_names=("a", "b"), shap_values=(0.4, -0.1), base_value=0.1
        )
    ]

    risk_report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        rule_report,
        anomaly_report,
        ml_report,
        shap_explanations,
        EntityIdentificationReport(),
    )
    standalone_explanation = GenerateExplanationUseCase().execute(
        rule_report, anomaly_report, ml_report, shap_explanations
    )

    assert risk_report.explanation.reasons == standalone_explanation.reasons
    assert risk_report.explanation.top_features == standalone_explanation.top_features
