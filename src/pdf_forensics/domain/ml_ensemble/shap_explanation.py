"""Raw per-feature local SHAP contribution for one specific model prediction.

`feature_names` and `shap_values` are parallel tuples (same order, same
length). A future Explainability module turns this into "top N features"
text; this type only carries the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShapExplanation:
    model_id: str
    feature_names: tuple[str, ...]
    shap_values: tuple[float, ...]
    base_value: float
