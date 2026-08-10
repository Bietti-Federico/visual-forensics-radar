"""Checks one document against the `LearnedInvariant`s
`FitEntityInvariantsUseCase` mined for its predicted entity, producing the
same `RuleFinding` shape the retired hand-written `entity_template_mismatch`
rule did — same `rule_id`, same severity, same confidence gate — so nothing
downstream (`GenerateRiskReportUseCase`, `GenerateExplanationUseCase`,
docs referencing the rule by id) needs to change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pdf_forensics.application.entity_invariants.fit_entity_invariants_use_case import (
    HISTOGRAM_KEY_SEPARATOR,
)
from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding

#: Below this, "doesn't confidently match any known entity" is already
#: covered by Risk Report's `entity_consistency` component — this check
#: only fires on a specific, confident, verifiable contradiction.
_MIN_CONFIDENCE = 0.6
_RULE_ID = "entity_template_mismatch"

_MISSING = object()


class CheckEntityInvariantsUseCase:
    def __init__(self, invariants: Sequence[LearnedInvariant]) -> None:
        self._invariants = tuple(invariants)

    def execute(
        self, feature_set: FeatureSet, entity_report: EntityIdentificationReport
    ) -> RuleFinding | None:
        if not self._invariants or not entity_report.predictions:
            return None

        prediction = entity_report.predictions[0]
        if prediction.confidence < _MIN_CONFIDENCE:
            return None

        violations = [
            violation
            for violation in (self._check(feature_set, inv) for inv in self._invariants)
            if violation is not None
        ]
        if not violations:
            return None

        return RuleFinding(
            rule_id=_RULE_ID,
            severity=AnomalySeverity.WARNING,
            confidence=prediction.confidence,
            explanation=(
                f"Identified as {prediction.predicted_entity} "
                f"(confidence={prediction.confidence:.0%}), but this document "
                f"contradicts every real sample seen: {'; '.join(violations)}."
            ),
            references=("docs/DOCUMENTACION.md",),
        )

    def _check(self, feature_set: FeatureSet, invariant: LearnedInvariant) -> str | None:
        actual = self._resolve(feature_set, invariant.feature_name)
        if actual is _MISSING or actual == invariant.expected_value:
            return None
        return f"{invariant.feature_name} is {actual!r} (expected {invariant.expected_value!r})"

    def _resolve(self, feature_set: FeatureSet, feature_name: str) -> Any:
        if HISTOGRAM_KEY_SEPARATOR in feature_name:
            base_name, key = feature_name.split(HISTOGRAM_KEY_SEPARATOR, 1)
            feature = feature_set.by_name(base_name)
            if feature is None or not isinstance(feature.value, Mapping):
                return _MISSING
            return feature.value.get(key, 0)

        feature = feature_set.by_name(feature_name)
        return _MISSING if feature is None else feature.value
