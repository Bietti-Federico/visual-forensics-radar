"""Small shared helper for feature-extractor tests."""

from __future__ import annotations

from pdf_forensics.domain.features.feature import Feature


def get_feature(features: list[Feature], name: str) -> Feature:
    for feature in features:
        if feature.name == name:
            return feature
    raise AssertionError(f"no feature named {name!r} among {[f.name for f in features]}")
