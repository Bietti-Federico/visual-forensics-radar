"""The complete set of findings produced by one rule-engine run."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding


@dataclass(slots=True)
class RuleEvaluationReport:
    findings: list[RuleFinding] = field(default_factory=list)

    def by_severity(self, severity: AnomalySeverity) -> list[RuleFinding]:
        return [finding for finding in self.findings if finding.severity is severity]

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[RuleFinding]:
        return iter(self.findings)
