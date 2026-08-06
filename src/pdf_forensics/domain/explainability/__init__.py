"""Domain model for Explainability: turning Modules 4-6's reports into human-readable output."""

from pdf_forensics.domain.explainability.explanation_report import ExplanationReport
from pdf_forensics.domain.explainability.top_feature_contribution import TopFeatureContribution

__all__ = ["ExplanationReport", "TopFeatureContribution"]
