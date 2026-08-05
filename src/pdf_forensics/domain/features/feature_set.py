"""The complete set of features extracted from one `PdfDocument`."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pdf_forensics.domain.features.enums import FeatureCategory
from pdf_forensics.domain.features.feature import Feature


@dataclass(slots=True)
class FeatureSet:
    """An ordered collection of `Feature`s, with grouping/lookup helpers.

    `to_dict()` flattens to a flat name->value mapping for a feature store or
    CSV export. Two features sharing a name is a bug in how the extractors
    were wired (not a legitimate case of "the later one wins") — `to_dict()`
    raises rather than silently dropping data a caller might rely on.
    """

    features: list[Feature] = field(default_factory=list)

    def by_category(self, category: FeatureCategory) -> list[Feature]:
        return [feature for feature in self.features if feature.category is category]

    def by_name(self, name: str) -> Feature | None:
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for feature in self.features:
            if feature.name in result:
                raise ValueError(f"Duplicate feature name: {feature.name!r}")
            result[feature.name] = feature.value
        return result

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self.features)
