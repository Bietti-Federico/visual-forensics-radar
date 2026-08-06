"""Flags a document saved more times than expected.

Default threshold of 2 ports the legacy tool's baseline observation
(`backend/pdf_metadata_extractor.py`): two revisions is the normal case for,
e.g., a signed receipt (one render, one save that adds the signature) —
three or more is the case worth a human look.
"""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding

_DEFAULT_THRESHOLD = 2


class UnexpectedIncrementalUpdateRule:
    rule_id = "unexpected_incremental_update"

    def __init__(self, threshold: int = _DEFAULT_THRESHOLD) -> None:
        self._threshold = threshold

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        revision_feature = feature_set.by_name("incremental_updates.revision_count")
        if revision_feature is None or not isinstance(revision_feature.value, int):
            return None
        revision_count = revision_feature.value
        if revision_count <= self._threshold:
            return None

        return RuleFinding(
            rule_id=self.rule_id,
            severity=AnomalySeverity.WARNING,
            confidence=1.0,
            explanation=(
                f"Document has {revision_count} revisions, more than the expected "
                f"baseline of {self._threshold}."
            ),
            references=("ISO 32000-1 §7.5.6",),
        )
