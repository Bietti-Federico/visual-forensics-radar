"""Shared `Feature` construction helper, used by every extractor in this package.

Not part of the public plugin API (leading underscore) — extractors are the
unit of independence the platform cares about, not this small piece of
boilerplate they happen to share.
"""

from __future__ import annotations

from types import MappingProxyType

from pdf_forensics.application.feature_extraction.extract_features_use_case import SCHEMA_VERSION
from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature, FeatureValue


def make_feature(
    category: FeatureCategory,
    name: str,
    value: FeatureValue,
    value_type: FeatureType,
    description: str,
    source: str,
    *,
    confidence: float = 1.0,
) -> Feature:
    # `Feature` is frozen/slotted and documents its compound values as
    # `Mapping`-based specifically so instances stay hashable — a plain
    # `dict` breaks that promise silently (no error until something actually
    # hashes the feature), so histogram-style extractors handing in a mutable
    # dict get it wrapped here, once, for every caller.
    if isinstance(value, dict):
        value = MappingProxyType(value)
    return Feature(
        name=name,
        value=value,
        value_type=value_type,
        description=description,
        source=source,
        confidence=confidence,
        category=category,
        schema_version=SCHEMA_VERSION,
    )
