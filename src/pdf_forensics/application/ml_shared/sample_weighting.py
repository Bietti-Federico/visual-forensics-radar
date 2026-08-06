"""Shared sample-weighting for Module 5 (Anomaly Detection) and Module 6 (ML Ensemble) fit/train.

Implemented via integer duplication rather than each plugin's own native
`sample_weight` support, which is inconsistent across plugins (e.g.
scikit-learn's `LocalOutlierFactor.fit` has no `sample_weight` parameter at
all). Duplicating a sample `round(weight)` times is a plugin-agnostic
substitute: a tree ensemble's bagging step and density-based detectors
(`LocalOutlierFactor`, `OneClassSVM`) both respond to duplicated points by
treating that region as denser/more influential — exactly what "trust this
sample more" should mean, with zero changes needed to any plugin.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def apply_sample_weight(items: Sequence[T], weights: Sequence[float] | None) -> list[T]:
    if weights is None:
        return list(items)
    if len(weights) != len(items):
        raise ValueError(
            f"sample_weight has {len(weights)} entries, expected {len(items)} (one per item)."
        )

    expanded: list[T] = []
    for item, weight in zip(items, weights, strict=True):
        expanded.extend([item] * max(0, round(weight)))
    return expanded
