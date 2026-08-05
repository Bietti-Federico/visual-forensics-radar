"""Shared `Feature` construction helper, used by every extractor in this package.

Not part of the public plugin API (leading underscore) — extractors are the
unit of independence the platform cares about, not this small piece of
boilerplate they happen to share.
"""

from __future__ import annotations

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
