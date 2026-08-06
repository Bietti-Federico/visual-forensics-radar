"""One feature's SHAP contribution, kept from a specific model's own explanation.

Not merged across models: different SHAP explainers (`TreeExplainer` vs.
`LinearExplainer`) operate in different output spaces, so ranking their raw
`shap_value` magnitudes against each other would imply a comparability that
doesn't exist. `model_id` says which model this contribution came from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopFeatureContribution:
    model_id: str
    feature_name: str
    shap_value: float
