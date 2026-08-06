"""Application boundary for running the rule engine over an already-extracted FeatureSet."""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.rule_engine.ports import RulePlugin
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport


class EvaluateRulesUseCase:
    def __init__(self, rules: Sequence[RulePlugin]) -> None:
        self._rules = tuple(rules)

    def execute(self, feature_set: FeatureSet) -> RuleEvaluationReport:
        findings = [
            finding
            for finding in (rule.evaluate(feature_set) for rule in self._rules)
            if finding is not None
        ]
        return RuleEvaluationReport(findings=findings)
