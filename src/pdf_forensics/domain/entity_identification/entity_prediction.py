"""One classifier's entity prediction for a document.

`probabilities` covers every entity the classifier was trained on — kept
alongside `predicted_entity`/`confidence` so callers can see how close the
runner-up was, not just the winner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntityPrediction:
    classifier_id: str
    predicted_entity: str
    confidence: float
    probabilities: Mapping[str, float]
