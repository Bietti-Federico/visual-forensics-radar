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
from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)
from pdf_forensics.domain.signature_verification.signature_verification_result import (
    SignatureVerificationResult,
)
from tests.fixtures.feature_helpers import build_feature_set


def _finding(severity: AnomalySeverity) -> RuleFinding:
    return RuleFinding(
        rule_id="test_rule", severity=severity, confidence=1.0, explanation="e", references=()
    )


def _signature_result(
    digest_intact: bool = True,
    cryptographically_valid: bool = True,
    coverage: SignatureCoverage = SignatureCoverage.ENTIRE_FILE,
) -> SignatureVerificationResult:
    return SignatureVerificationResult(
        field_name="Signature1",
        digest_intact=digest_intact,
        cryptographically_valid=cryptographically_valid,
        coverage=coverage,
        signer_subject="Test Signer",
        signing_time=None,
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
        SignatureVerificationReport(),
    )
    assert report.risk_score == 0
    assert {c.name: c.score for c in report.components} == {
        "rule_engine": 0.0,
        "ml_probability": 0.0,
        "anomaly_detection": 0.0,
        "structural": 0.0,
        "metadata": 0.0,
        "entity_consistency": 0.0,
        "signature_integrity": 0.0,
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
        SignatureVerificationReport(),
    )
    rule_engine_component = next(c for c in report.components if c.name == "rule_engine")
    assert rule_engine_component.score == 0.9
    assert report.risk_score == round(0.9 * 0.24 * 100)


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
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
    )
    anomaly_component = next(c for c in report.components if c.name == "anomaly_detection")
    assert anomaly_component.score == 0.5


def test_anomaly_detection_score_is_fraction_of_detectors_flagged() -> None:
    # A single detector flagging out of four (e.g. one_class_svm sitting
    # exactly on its own decision boundary — a known artifact with small
    # per-entity training batches) shouldn't score anywhere near as high as
    # most of the ensemble agreeing.
    one_flagged = AnomalyDetectionReport(
        scores=[
            AnomalyScore(detector_id="a", score=0.0, is_anomaly=True),
            AnomalyScore(detector_id="b", score=1.0, is_anomaly=False),
            AnomalyScore(detector_id="c", score=1.0, is_anomaly=False),
            AnomalyScore(detector_id="d", score=1.0, is_anomaly=False),
        ]
    )
    three_flagged = AnomalyDetectionReport(
        scores=[
            AnomalyScore(detector_id="a", score=5.0, is_anomaly=True),
            AnomalyScore(detector_id="b", score=5.0, is_anomaly=True),
            AnomalyScore(detector_id="c", score=5.0, is_anomaly=True),
            AnomalyScore(detector_id="d", score=1.0, is_anomaly=False),
        ]
    )

    def _score(anomaly_report: AnomalyDetectionReport) -> float:
        report = GenerateRiskReportUseCase().execute(
            _clean_feature_set(),
            RuleEvaluationReport(),
            anomaly_report,
            MlEnsembleReport(),
            [],
            EntityIdentificationReport(),
            SignatureVerificationReport(),
        )
        return next(c for c in report.components if c.name == "anomaly_detection").score

    assert _score(one_flagged) == 0.25
    assert _score(three_flagged) == 0.75


def test_structural_score_zero_when_features_absent() -> None:
    report = GenerateRiskReportUseCase().execute(
        build_feature_set({}),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
    )
    custom_weights = RiskWeights(
        rule_engine=0.7,
        ml_probability=0.04,
        anomaly_detection=0.04,
        structural=0.04,
        metadata=0.04,
        entity_consistency=0.04,
        signature_integrity=0.1,
    )
    custom_report = GenerateRiskReportUseCase(weights=custom_weights).execute(
        _clean_feature_set(),
        rule_report,
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        SignatureVerificationReport(),
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
        SignatureVerificationReport(),
    )
    entity_component = next(c for c in report.components if c.name == "entity_consistency")
    assert entity_component.score == pytest.approx(0.3)


def test_signature_score_zero_when_no_signature_present() -> None:
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        SignatureVerificationReport(),
    )
    signature_component = next(c for c in report.components if c.name == "signature_integrity")
    assert signature_component.score == 0.0


def test_signature_score_is_max_when_digest_broken() -> None:
    signature_report = SignatureVerificationReport(results=[_signature_result(digest_intact=False)])
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        signature_report,
    )
    signature_component = next(c for c in report.components if c.name == "signature_integrity")
    assert signature_component.score == 1.0


def test_signature_score_is_moderate_for_partial_coverage() -> None:
    signature_report = SignatureVerificationReport(
        results=[_signature_result(coverage=SignatureCoverage.PARTIAL)]
    )
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        signature_report,
    )
    signature_component = next(c for c in report.components if c.name == "signature_integrity")
    assert signature_component.score == 0.7


def test_signature_score_zero_for_fully_valid_signature() -> None:
    signature_report = SignatureVerificationReport(results=[_signature_result()])
    report = GenerateRiskReportUseCase().execute(
        _clean_feature_set(),
        RuleEvaluationReport(),
        AnomalyDetectionReport(),
        MlEnsembleReport(),
        [],
        EntityIdentificationReport(),
        signature_report,
    )
    signature_component = next(c for c in report.components if c.name == "signature_integrity")
    assert signature_component.score == 0.0


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
        SignatureVerificationReport(),
    )
    standalone_explanation = GenerateExplanationUseCase().execute(
        rule_report, anomaly_report, ml_report, shap_explanations
    )

    assert risk_report.explanation.reasons == standalone_explanation.reasons
    assert risk_report.explanation.top_features == standalone_explanation.top_features
