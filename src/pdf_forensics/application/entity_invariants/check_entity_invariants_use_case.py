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

        # A minority of incidental violations is treated the same as
        # before (WARNING) — with few learned invariants (see
        # TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY=1), a couple of
        # coincidental mismatches don't mean the document is fake. But a
        # document contradicting most or all of an entity's invariants at
        # once isn't "structurally a bit off" — every dimension the real
        # template is consistent on disagrees, which in practice means
        # this isn't that entity's template at all, so it's scored as
        # CRITICAL like the strongest structural findings elsewhere.
        severity = (
            AnomalySeverity.CRITICAL
            if len(violations) > len(self._invariants) / 2
            else AnomalySeverity.WARNING
        )

        return RuleFinding(
            rule_id=_RULE_ID,
            severity=severity,
            confidence=prediction.confidence,
            explanation=(
                f"Identificado como {prediction.predicted_entity} "
                f"(confianza={prediction.confidence:.0%}), pero este documento "
                f"contradice lo observado en todos los documentos reales: "
                f"{'; '.join(violations)}."
            ),
            references=("docs/DOCUMENTACION.md",),
        )

    def _check(self, feature_set: FeatureSet, invariant: LearnedInvariant) -> str | None:
        actual = self._resolve(feature_set, invariant.feature_name)
        if actual is _MISSING or actual == invariant.expected_value:
            return None
        return f"{invariant.feature_name}={actual!r} (se esperaba {invariant.expected_value!r})"

    def _resolve(self, feature_set: FeatureSet, feature_name: str) -> Any:
        if HISTOGRAM_KEY_SEPARATOR in feature_name:
            base_name, key = feature_name.split(HISTOGRAM_KEY_SEPARATOR, 1)
            feature = feature_set.by_name(base_name)
            if feature is None or not isinstance(feature.value, Mapping):
                return _MISSING
            return feature.value.get(key, 0)

        feature = feature_set.by_name(feature_name)
        return _MISSING if feature is None else feature.value
