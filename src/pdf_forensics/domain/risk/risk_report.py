"""The final deliverable of the pipeline: a 0-100 risk score plus its full explanation.

Embeds Module 7's `ExplanationReport` directly, matching the platform brief's
own illustrative example, which shows the score and its reasons/SHAP top
features together as one output.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.risk.risk_component_score import RiskComponentScore


@dataclass(frozen=True, slots=True)
class RiskReport:
    risk_score: int
    components: tuple[RiskComponentScore, ...]
    explanation: ExplanationReport
