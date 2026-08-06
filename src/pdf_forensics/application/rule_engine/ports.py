"""The port `EvaluateRulesUseCase` depends on; `plugins/rules/` implements it.

A rule is a declarative statement about specific *named* features (e.g. "if
`metadata.has_info_dict` is `False`...") — unlike `feature_extraction`'s port,
where name-string coupling was deliberately avoided (Module 3's fingerprint),
here that coupling is the entire point of the abstraction: a rule exists to
assert something about a named signal.
"""

from __future__ import annotations

from typing import Protocol

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class RulePlugin(Protocol):
    rule_id: str

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        """Return a `RuleFinding` if this rule triggers, or `None` if it doesn't apply.

        `None` covers both "nothing wrong" and "the features needed to
        evaluate this rule aren't present/parseable" — a rule never guesses.
        """
        ...
