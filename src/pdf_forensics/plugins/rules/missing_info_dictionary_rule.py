"""Flags a document with no /Info dictionary reachable from the trailer."""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class MissingInfoDictionaryRule:
    rule_id = "missing_info_dictionary"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        has_info = feature_set.by_name("metadata.has_info_dict")
        if has_info is None or has_info.value is not False:
            return None

        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.WARNING,
            confidence=1.0,
            explanation="El trailer no tiene una entrada /Info que resuelva a un diccionario.",
            references=("ISO 32000-1 §14.3.3",),
        )
