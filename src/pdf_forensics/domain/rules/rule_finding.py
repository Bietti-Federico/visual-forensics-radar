"""One triggered rule finding: what fired, how severe, how confident, and why."""

from __future__ import annotations

from dataclasses import dataclass

from pdf_forensics.domain.pdf.anomalies import AnomalySeverity


@dataclass(frozen=True, slots=True)
class RuleFinding:
    rule_id: str
    severity: AnomalySeverity
    confidence: float
    explanation: str
    references: tuple[str, ...]
