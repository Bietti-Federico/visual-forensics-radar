"""One structural fact found identical across every genuine training sample
seen for an entity, mined automatically rather than hand-written.

`feature_name` is either a real `Feature.name` (for boolean features) or a
synthetic `<feature_name>::<key>` name for one key of a `DICT`-valued
feature (e.g. `streams.filter_histogram::DCTDecode`) — see
`application/entity_invariants/fit_entity_invariants_use_case.py` for how
the latter is derived.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf_forensics.domain.features.feature import FeatureValue


@dataclass(frozen=True, slots=True)
class LearnedInvariant:
    feature_name: str
    expected_value: FeatureValue
