"""Selects the numeric subset of a FeatureSet for feeding to any ML model.

Shared by both `anomaly_detection` (unsupervised) and `ml_ensemble`
(supervised) — nothing about numeric-feature selection is specific to either,
so it lives once here instead of as two copies that could drift.

Only `INTEGER`/`FLOAT`/`BOOLEAN`-typed features are vectorizable this
iteration — `DICT`/`LIST`/`STRING` features (histograms, producer strings,
...) are explicitly out of scope, not silently dropped without explanation.
A numeric-typed feature whose actual value is `None` (e.g. `catalog.page_count`
when `/Pages` doesn't resolve) is *omitted* rather than coerced to `0.0` —
"unknown" and "zero" are different facts, and coercing would tell every
model that an unresolvable page tree looks identical to a zero-page one.
"""

from __future__ import annotations

from pdf_forensics.domain.features.enums import FeatureType
from pdf_forensics.domain.features.feature_set import FeatureSet

_VECTORIZABLE_TYPES = (FeatureType.INTEGER, FeatureType.FLOAT, FeatureType.BOOLEAN)


def select_numeric_features(feature_set: FeatureSet) -> dict[str, float]:
    vector: dict[str, float] = {}
    for feature in feature_set:
        if feature.value_type not in _VECTORIZABLE_TYPES or feature.value is None:
            continue
        vector[feature.name] = float(feature.value)  # type: ignore[arg-type]
    return vector
