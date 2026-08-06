"""Flags xref-related structural anomalies recorded during parsing.

Severity mirrors what Module 1 already assigned to these exact anomaly
codes, so the same underlying fact is never reported at two different
severities across the platform: `xref_unparseable_fallback_used` is
`CRITICAL` (the xref itself was unusable); `xref_offset_mismatch` and
`broken_prev_chain` are `WARNING`.
"""

from __future__ import annotations

from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalyCode, AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding

_CRITICAL_CODE = AnomalyCode.XREF_UNPARSEABLE_FALLBACK_USED.value
_WARNING_CODES = (AnomalyCode.XREF_OFFSET_MISMATCH.value, AnomalyCode.BROKEN_PREV_CHAIN.value)


class BrokenXrefRule:
    rule_id = "broken_xref"

    def evaluate(self, feature_set: FeatureSet) -> RuleFinding | None:
        histogram_feature = feature_set.by_name("statistics.anomaly_code_histogram")
        if histogram_feature is None or not isinstance(histogram_feature.value, dict):
            return None
        histogram = histogram_feature.value

        if histogram.get(_CRITICAL_CODE, 0) > 0:
            return self._finding(AnomalySeverity.CRITICAL, "the xref could not be parsed at all")

        triggered = [code for code in _WARNING_CODES if histogram.get(code, 0) > 0]
        if not triggered:
            return None
        return self._finding(AnomalySeverity.WARNING, ", ".join(triggered))

    def _finding(self, severity: AnomalySeverity, reason: str) -> RuleFinding:
        return RuleFinding(
            rule_id=self.rule_id,
            severity=severity,
            confidence=1.0,
            explanation=f"Cross-reference anomalies detected: {reason}.",
            references=("ISO 32000-1 §7.5.4",),
        )
