"""One supervised model's prediction for a document.

`probability` is standardized as P(manipulated) — i.e. P(is_original ==
False) — so "higher = more suspicious" holds across this module and Module
5's "higher = more anomalous," keeping the platform's score conventions
consistent everywhere a score appears.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    model_id: str
    probability: float
    #: The model's own predict() decision — not just a >0.5 threshold on
    #: `probability`, since predict()/predict_proba() can disagree slightly
    #: at the decision boundary; this reports what the model actually decided.
    predicted_label: bool
