"""Small shared helpers for feature-extractor and rule-engine tests."""

from __future__ import annotations

from typing import Any

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType
from pdf_forensics.domain.features.feature import Feature
from pdf_forensics.domain.features.feature_set import FeatureSet


def get_feature(features: list[Feature], name: str) -> Feature:
    for feature in features:
        if feature.name == name:
            return feature
    raise AssertionError(f"no feature named {name!r} among {[f.name for f in features]}")


def _infer_value_type(value: Any) -> FeatureType:
    # bool is a subclass of int, so it must be checked first.
    if isinstance(value, bool):
        return FeatureType.BOOLEAN
    if isinstance(value, int):
        return FeatureType.INTEGER
    if isinstance(value, float):
        return FeatureType.FLOAT
    if isinstance(value, dict):
        return FeatureType.DICT
    if isinstance(value, list | tuple):
        return FeatureType.LIST
    return FeatureType.STRING


def build_feature_set(values: dict[str, Any]) -> FeatureSet:
    """A minimal `FeatureSet` for rule tests: name -> value, everything else dummied out."""
    return FeatureSet(
        features=[
            Feature(
                name=name,
                value=value,
                value_type=_infer_value_type(value),
                description="test feature",
                source="test",
                confidence=1.0,
                category=FeatureCategory.GENERAL,
                schema_version="1.0.0",
            )
            for name, value in values.items()
        ]
    )
