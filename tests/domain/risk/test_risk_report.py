from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.risk.risk_component_score import RiskComponentScore
from pdf_forensics.domain.risk.risk_report import RiskReport


def test_fields() -> None:
    component = RiskComponentScore(name="rule_engine", score=0.5, weight=0.3)
    explanation = ExplanationReport(reasons=["a reason"])
    report = RiskReport(risk_score=42, components=(component,), explanation=explanation)

    assert report.risk_score == 42
    assert report.components == (component,)
    assert report.explanation is explanation
