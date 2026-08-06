"""Flags a document confidently identified as a known entity but structurally
inconsistent with every real document of that entity sampled so far.

Comparing real documents from three entities (see docs/entity-identification.md)
turned up two perfect, entity-specific invariants:

- `catalog.has_acroform`: `True` for every real ANSES document sampled,
  `False` for every real Municipalidad de La Rioja / Jujuy document sampled
  — consistent with ANSES receipts' visible "Firmado Digitalmente" badge,
  backed by an actual /AcroForm digital-signature structure, not just text.
- Embedded JPEG image count (`streams.filter_histogram["DCTDecode"]`):
  exactly 0 for ANSES, 1 for La Rioja, 2 for Jujuy, in every real document of
  that entity sampled — likely each entity's fixed letterhead/seal artwork.

Scope: a fixed list of required invariants per entity, checked only when the
classifier's confidence clears `_MIN_CONFIDENCE`. Below that threshold,
"doesn't confidently match any known entity" is exactly what Risk Report's
`entity_consistency` component already covers — this rule only fires on a
specific, confident, verifiable contradiction. Extend
`_ENTITY_TEMPLATE_INVARIANTS` as more entities/invariants are found.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pdf_forensics.domain.entity_identification.entity_identification_report import (
    EntityIdentificationReport,
)
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.pdf.anomalies import AnomalySeverity
from pdf_forensics.domain.rules.rule_finding import RuleFinding

_MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class _TemplateInvariant:
    feature_name: str
    is_expected: Callable[[Any], bool]
    rationale: str


def _dct_count_is(count: int) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, dict) and value.get("DCTDecode", 0) == count


_ENTITY_TEMPLATE_INVARIANTS: dict[str, tuple[_TemplateInvariant, ...]] = {
    "ANSES": (
        _TemplateInvariant(
            "catalog.has_acroform",
            lambda value: value is True,
            "every real ANSES document sampled has an /AcroForm entry "
            "(the digital-signature structure behind its 'Firmado Digitalmente' badge)",
        ),
        _TemplateInvariant(
            "streams.filter_histogram",
            _dct_count_is(0),
            "no real ANSES document sampled embeds a JPEG (/DCTDecode) image",
        ),
    ),
    "LA_RIOJA": (
        _TemplateInvariant(
            "catalog.has_acroform",
            lambda value: value is False,
            "no real Municipalidad de La Rioja document sampled has an /AcroForm entry",
        ),
        _TemplateInvariant(
            "streams.filter_histogram",
            _dct_count_is(1),
            "every real Municipalidad de La Rioja document sampled embeds exactly "
            "one JPEG (/DCTDecode) image",
        ),
    ),
    "JUJUY": (
        _TemplateInvariant(
            "catalog.has_acroform",
            lambda value: value is False,
            "no real Municipalidad de Jujuy document sampled has an /AcroForm entry",
        ),
        _TemplateInvariant(
            "streams.filter_histogram",
            _dct_count_is(2),
            "every real Municipalidad de Jujuy document sampled embeds exactly "
            "two JPEG (/DCTDecode) images",
        ),
    ),
}


class EntityTemplateMismatchRule:
    rule_id = "entity_template_mismatch"

    def evaluate(
        self, feature_set: FeatureSet, entity_report: EntityIdentificationReport
    ) -> RuleFinding | None:
        for prediction in entity_report:
            invariants = _ENTITY_TEMPLATE_INVARIANTS.get(prediction.predicted_entity)
            if invariants is None or prediction.confidence < _MIN_CONFIDENCE:
                continue

            violated = [
                invariant.rationale
                for invariant in invariants
                if self._violates(feature_set, invariant)
            ]
            if not violated:
                continue

            return RuleFinding(
                rule_id=self.rule_id,
                severity=AnomalySeverity.WARNING,
                confidence=prediction.confidence,
                explanation=(
                    f"Identified as {prediction.predicted_entity} "
                    f"(confidence={prediction.confidence:.0%}), but this document "
                    f"contradicts every real sample seen: {'; '.join(violated)}."
                ),
                references=("docs/entity-identification.md",),
            )
        return None

    def _violates(self, feature_set: FeatureSet, invariant: _TemplateInvariant) -> bool:
        feature = feature_set.by_name(invariant.feature_name)
        if feature is None:
            return False
        return not invariant.is_expected(feature.value)
