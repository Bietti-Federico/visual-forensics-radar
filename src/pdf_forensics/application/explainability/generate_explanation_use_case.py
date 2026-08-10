"""Turns Modules 4-6's reports into a human-readable ExplanationReport.

A pure function of four already-computed reports — no dependency on *how*
they were produced (no import of Modules 4/5/6's plugins, only their domain
types), and no new reasons invented beyond what those reports already say.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.explainability.top_feature_contribution import TopFeatureContribution
from pdf_forensics.domain.ml_ensemble.ml_ensemble_report import MlEnsembleReport
from pdf_forensics.domain.ml_ensemble.shap_explanation import ShapExplanation
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport

_DEFAULT_TOP_N_FEATURES = 5
_SEVERITY_RANK = {
    AnomalySeverity.CRITICAL: 0,
    AnomalySeverity.WARNING: 1,
    AnomalySeverity.INFO: 2,
}

#: Plain-language labels for the bounded, fixed set of detector/model ids
#: this platform ships (`plugins/anomaly_detection/`, `plugins/ml_ensemble/`)
#: — an unrecognized id (there shouldn't be one) still displays as-is.
_DETECTOR_LABELS = {
    "isolation_forest": "el detector de aislamiento (Isolation Forest)",
    "one_class_svm": "el detector de frontera (One-Class SVM)",
    "local_outlier_factor": "el detector de densidad local (Local Outlier Factor)",
    "autoencoder": "el autoencoder",
}
_MODEL_LABELS = {
    "random_forest": "el modelo de bosque aleatorio (Random Forest)",
    "extra_trees": "el modelo de árboles extra (Extra Trees)",
    "logistic_regression": "el modelo de regresión logística",
    "xgboost": "el modelo XGBoost",
    "stacking_ensemble": "el ensamble por apilado (Stacking)",
    "voting_ensemble": "el ensamble por votación (Voting)",
}


class GenerateExplanationUseCase:
    def __init__(self, top_n_features: int = _DEFAULT_TOP_N_FEATURES) -> None:
        self._top_n_features = top_n_features

    def execute(
        self,
        rule_report: RuleEvaluationReport,
        anomaly_report: AnomalyDetectionReport,
        ml_report: MlEnsembleReport,
        shap_explanations: Sequence[ShapExplanation],
    ) -> ExplanationReport:
        return ExplanationReport(
            reasons=self._build_reasons(rule_report, anomaly_report, ml_report),
            top_features=self._build_top_features(shap_explanations),
        )

    def _build_reasons(
        self,
        rule_report: RuleEvaluationReport,
        anomaly_report: AnomalyDetectionReport,
        ml_report: MlEnsembleReport,
    ) -> list[str]:
        reasons = [
            finding.explanation
            for finding in sorted(rule_report, key=lambda f: _SEVERITY_RANK[f.severity])
        ]
        reasons.extend(
            f"{_DETECTOR_LABELS.get(score.detector_id, score.detector_id)} marcó este "
            f"documento como fuera de lo normal para esta entidad "
            f"(score interno={score.score:.3f}, no comparable entre detectores)."
            for score in anomaly_report
            if score.is_anomaly
        )
        reasons.extend(
            f"{_MODEL_LABELS.get(prediction.model_id, prediction.model_id)} predice que "
            f"este documento está manipulado (probabilidad={prediction.probability:.0%})."
            for prediction in ml_report
            if prediction.predicted_label
        )
        return reasons

    def _build_top_features(
        self, shap_explanations: Sequence[ShapExplanation]
    ) -> list[TopFeatureContribution]:
        top_features: list[TopFeatureContribution] = []
        for explanation in shap_explanations:
            paired = sorted(
                zip(explanation.feature_names, explanation.shap_values, strict=True),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )
            top_features.extend(
                TopFeatureContribution(
                    model_id=explanation.model_id, feature_name=name, shap_value=value
                )
                for name, value in paired[: self._top_n_features]
            )
        return top_features
