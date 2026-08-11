"""Flags a document with at least one duplicate object id definition."""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding


class DuplicateObjectIdsRule:
    rule_id = "duplicate_object_ids"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        count_feature = feature_set.by_name("objects.duplicate_object_id_count")
        if count_feature is None or not isinstance(count_feature.value, int):
            return None
        count = count_feature.value
        if count <= 0:
            return None

        verbo = "está definido" if count == 1 else "están definidos"
        sustantivo = "número de objeto" if count == 1 else "números de objeto"
        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.WARNING,
            confidence=1.0,
            explanation=(f"{count} {sustantivo} {verbo} más de una vez en este archivo."),
            references=("ISO 32000-1 §7.5.4",),
        )
