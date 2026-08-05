"""A single named, typed, documented forensic feature value.

Every feature a plugin produces carries not just a value but where it came
from and how confident that reading is — a feature store or report should
never need to go back to the extractor's source code to answer "what does
this column mean" or "how was this computed."
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pdf_forensics.domain.features.enums import FeatureCategory, FeatureType

#: Compound values use `tuple`/`Mapping` (not `list`/`dict`) so `Feature`
#: stays frozen and hashable.
FeatureValue = bool | int | float | str | tuple[Any, ...] | Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class Feature:
    """One extracted feature.

    `source` is free-text provenance (e.g. `"PdfDocument.trailer"`,
    `"catalog./Root/Pages/Count"`) so a report can cite exactly which field
    produced a value. `confidence` is 0.0-1.0: `1.0` for a direct structural
    read (everything this module produces), lower only for genuinely
    derived/heuristic values a later module might add.
    """

    name: str
    value: FeatureValue
    value_type: FeatureType
    description: str
    source: str
    confidence: float
    category: FeatureCategory
    schema_version: str
