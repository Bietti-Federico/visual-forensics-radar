"""Flags a trailer with no /Root entry.

Severity matches Module 1's `TRAILER_MISSING_ROOT` anomaly for the same
underlying fact.
"""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class TrailerMissingRootRule:
    rule_id = "trailer_missing_root"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        has_root = feature_set.by_name("trailer.has_root")
        if has_root is None or has_root.value is not False:
            return None

        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.WARNING,
            confidence=1.0,
            explanation="The trailer has no /Root entry.",
            references=("ISO 32000-1 §7.5.5",),
        )
