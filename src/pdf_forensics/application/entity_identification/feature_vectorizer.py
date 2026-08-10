"""Selects features for entity identification.

Every numeric structural feature (same set `ml_shared.select_numeric_features`
selects for Modules 5/6) plus the `/Producer` and `/Creator` metadata
strings, passed through raw rather than dropped. In practice the producer
string is the single strongest signal observed so far — each of the three
real-world templates analyzed uses a distinct, internally consistent
producer (`iTextSharp 5.5.13.4` / `iTextSharp 5.5.8` / `mPDF 5.7`), see
`docs/DOCUMENTACION.md`. `DictVectorizer` one-hot-encodes string values
automatically, so no separate encoding step is needed here.
"""

from __future__ import annotations

from pdf_forensics.application.ml_shared.feature_vectorizer import select_numeric_features
from pdf_forensics.domain.features.feature_set import FeatureSet

_STRING_FEATURE_NAMES = ("metadata.producer", "metadata.creator")


def select_entity_features(feature_set: FeatureSet) -> dict[str, float | str]:
    vector: dict[str, float | str] = dict(select_numeric_features(feature_set))
    for name in _STRING_FEATURE_NAMES:
        feature = feature_set.by_name(name)
        if feature is not None and feature.value is not None:
            vector[name] = str(feature.value)
    return vector
