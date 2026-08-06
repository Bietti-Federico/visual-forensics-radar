"""A human-readable explanation of a document's evaluation: reasons plus top SHAP features.

Not a risk score — a future Risk Report module combines this with a
weighted numeric score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pdf_forensics.domain.explainability.top_feature_contribution import TopFeatureContribution


@dataclass(slots=True)
class ExplanationReport:
    reasons: list[str] = field(default_factory=list)
    top_features: list[TopFeatureContribution] = field(default_factory=list)
