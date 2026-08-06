"""A second, narrower rule-plugin port: rules that need to know which entity
Entity Identification predicted, not just the FeatureSet.

Kept separate from `RulePlugin` (not a breaking signature change to it)
deliberately — the seven existing structural rules have no business knowing
about entity predictions, and forcing every rule to accept a parameter only
one of them uses would blur exactly the "rules assert something about named
features" boundary `ports.py`'s own docstring describes. `EvaluateRulesUseCase`
and `EvaluateEntityAwareRulesUseCase` both produce plain `RuleFinding`s into
the same `RuleEvaluationReport` type, so callers merge the two freely.
"""

from __future__ import annotations

from typing import Protocol

from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class EntityAwareRulePlugin(Protocol):
    rule_id: str

    def evaluate(
        self, feature_set: FeatureSet, entity_report: EntityIdentificationReport
    ) -> RuleFinding | None:
        """Return a `RuleFinding` if this rule triggers, or `None` if it doesn't apply."""
        ...
