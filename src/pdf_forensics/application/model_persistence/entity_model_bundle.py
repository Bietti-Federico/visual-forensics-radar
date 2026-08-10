"""One entity's slice of a persisted model bundle.

`detectors`/`models` are empty tuples (not `None`) when there wasn't enough
data to fit them at the last retrain — `()` composes naturally with
`DetectAnomaliesUseCase`/`PredictUseCase`, which already treat "no plugins
configured" as "nothing to report" (see their own module docstrings), no
extra `None`-checking needed at the call site.

`genuine_count`/`confirmed_fraud_count`/`ml_ensemble_ready` are a snapshot
from the retrain that produced this bundle — used to decide, per predicted
entity, whether `RiskWeights()` or `RiskWeights().without_ml_probability()`
applies (see `application/document_scoring/score_document_use_case.py`).

`invariants` is likewise `()` below its own minimum-sample threshold —
see `application/entity_invariants/fit_entity_invariants_use_case.py` and
`application/model_training/retrain_models_use_case.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf_forensics.application.anomaly_detection.ports import AnomalyDetectorPlugin
from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant


@dataclass(frozen=True, slots=True)
class EntityModelBundle:
    detectors: tuple[AnomalyDetectorPlugin, ...]
    models: tuple[SupervisedModelPlugin, ...]
    invariants: tuple[LearnedInvariant, ...]
    genuine_count: int
    confirmed_fraud_count: int
    ml_ensemble_ready: bool
