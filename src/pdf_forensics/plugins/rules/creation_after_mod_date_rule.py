"""Flags a document whose /CreationDate is after its /ModDate, once both are UTC-normalized."""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.pdf.pdf_date import parse_pdf_date
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class CreationAfterModDateRule:
    rule_id = "creation_after_mod_date"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        creation_feature = feature_set.by_name("metadata.creation_date")
        mod_feature = feature_set.by_name("metadata.mod_date")
        if creation_feature is None or mod_feature is None:
            return None

        creation_value = creation_feature.value
        mod_value = mod_feature.value
        creation = parse_pdf_date(creation_value if isinstance(creation_value, str) else None)
        mod = parse_pdf_date(mod_value if isinstance(mod_value, str) else None)
        if creation is None or mod is None or creation <= mod:
            return None

        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.WARNING,
            confidence=1.0,
            explanation=(
                f"La fecha de creación ({creation.isoformat()}) es posterior a la de "
                f"modificación ({mod.isoformat()}), normalizadas ambas a UTC."
            ),
            references=("ISO 32000-1 §14.3.3",),
        )
