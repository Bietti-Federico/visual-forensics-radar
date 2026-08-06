from pdf_forensics.application.rule_engine.evaluate_entity_aware_rules_use_case import (
    EvaluateEntityAwareRulesUseCase,
)
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding
from tests.fixtures.feature_helpers import build_feature_set


class _AlwaysTriggersRule:
    rule_id = "always_triggers"

    def evaluate(self, feature_set, entity_report):
        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.INFO,
            confidence=1.0,
            explanation="e",
            references=(),
        )


class _NeverTriggersRule:
    rule_id = "never_triggers"

    def evaluate(self, feature_set, entity_report):
        return None


def test_collects_findings_from_triggered_rules_only() -> None:
    use_case = EvaluateEntityAwareRulesUseCase([_AlwaysTriggersRule(), _NeverTriggersRule()])
    report = use_case.execute(build_feature_set({}), EntityIdentificationReport())

    assert len(report) == 1
    assert report.findings[0].rule_id == "always_triggers"


def test_empty_rules_produce_empty_report() -> None:
    report = EvaluateEntityAwareRulesUseCase([]).execute(
        build_feature_set({}), EntityIdentificationReport()
    )
    assert len(report) == 0
