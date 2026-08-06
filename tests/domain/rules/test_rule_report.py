from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport


def _finding(rule_id: str, severity: AnomalySeverity) -> RuleFinding:
    return RuleFinding(
        rule_id=rule_id, severity=severity, confidence=1.0, explanation="e", references=()
    )


def test_by_severity_filters_correctly() -> None:
    warning = _finding("a", AnomalySeverity.WARNING)
    critical = _finding("b", AnomalySeverity.CRITICAL)
    report = RuleEvaluationReport(findings=[warning, critical])

    assert report.by_severity(AnomalySeverity.WARNING) == [warning]
    assert report.by_severity(AnomalySeverity.CRITICAL) == [critical]
    assert report.by_severity(AnomalySeverity.INFO) == []


def test_len_and_iter() -> None:
    findings = [_finding("a", AnomalySeverity.WARNING), _finding("b", AnomalySeverity.INFO)]
    report = RuleEvaluationReport(findings=findings)
    assert len(report) == 2
    assert list(report) == findings


def test_empty_report_defaults() -> None:
    report = RuleEvaluationReport()
    assert len(report) == 0
    assert report.by_severity(AnomalySeverity.WARNING) == []
