"""Flags a document confidently identified as a known entity but structurally
inconsistent with every real document of that entity sampled so far.

Comparing real documents from three entities (see docs/entity-identification.md)
turned up one boolean feature that's a perfect, entity-specific invariant:
`catalog.has_acroform` is `True` for every real ANSES document sampled, and
`False` for every real Municipalidad de La Rioja / Jujuy document sampled —
consistent with ANSES receipts' visible "Firmado Digitalmente" badge, which
is backed by an actual /AcroForm digital-signature structure, not just text.

Scope: one required boolean feature per entity, checked only when the
classifier's confidence clears `_MIN_CONFIDENCE`. Below that threshold,
"doesn't confidently match any known entity" is exactly what Risk Report's
`entity_consistency` component already covers — this rule only fires on a
specific, confident, verifiable contradiction. Extend
`_ENTITY_TEMPLATE_REQUIREMENTS` as more entities/invariants are found.
"""

from __future__ import annotations

from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding

_MIN_CONFIDENCE = 0.6

# entity -> (feature_name, expected_value, human-readable rationale)
_ENTITY_TEMPLATE_REQUIREMENTS: dict[str, tuple[str, bool, str]] = {
    "ANSES": (
        "catalog.has_acroform",
        True,
        "every real ANSES document sampled has an /AcroForm entry "
        "(the digital-signature structure behind its 'Firmado Digitalmente' badge)",
    ),
    "LA_RIOJA": (
        "catalog.has_acroform",
        False,
        "no real Municipalidad de La Rioja document sampled has an /AcroForm entry",
    ),
    "JUJUY": (
        "catalog.has_acroform",
        False,
        "no real Municipalidad de Jujuy document sampled has an /AcroForm entry",
    ),
}


class EntityTemplateMismatchRule:
    rule_id = "entity_template_mismatch"

    def evaluate(
        self, feature_set: FeatureSet, entity_report: EntityIdentificationReport
    ) -> RuleFinding | None:
        for prediction in entity_report:
            requirement = _ENTITY_TEMPLATE_REQUIREMENTS.get(prediction.predicted_entity)
            if requirement is None or prediction.confidence < _MIN_CONFIDENCE:
                continue

            feature_name, expected_value, rationale = requirement
            feature = feature_set.by_name(feature_name)
            if feature is None or feature.value == expected_value:
                continue

            return RuleFinding(
                rule_id=self.rule_id,
                severity=AnomalySeverity.WARNING,
                confidence=prediction.confidence,
                explanation=(
                    f"Identified as {prediction.predicted_entity} "
                    f"(confidence={prediction.confidence:.0%}), but {rationale} — "
                    f"this document does not match."
                ),
                references=("docs/entity-identification.md",),
            )
        return None
