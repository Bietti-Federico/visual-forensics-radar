"""Application boundary for running entity-aware rules over an already-extracted
FeatureSet plus an EntityIdentificationReport.

Mirrors `EvaluateRulesUseCase` exactly, one parameter wider. Produces the
same `RuleEvaluationReport` type, so a caller merges this with the plain
Rule Engine's findings before handing them to Explainability/Risk Report —
neither of those needs to know entity-aware rules exist.
"""

from __future__ import annotations

from collections.abc import Sequence

from pdf_forensics.application.rule_engine.entity_aware_ports import EntityAwareRulePlugin
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.rules.rule_report import RuleEvaluationReport


class EvaluateEntityAwareRulesUseCase:
    def __init__(self, rules: Sequence[EntityAwareRulePlugin]) -> None:
        self._rules = tuple(rules)

    def execute(
        self, feature_set: FeatureSet, entity_report: EntityIdentificationReport
    ) -> RuleEvaluationReport:
        findings = [
            finding
            for finding in (rule.evaluate(feature_set, entity_report) for rule in self._rules)
            if finding is not None
        ]
        return RuleEvaluationReport(findings=findings)
