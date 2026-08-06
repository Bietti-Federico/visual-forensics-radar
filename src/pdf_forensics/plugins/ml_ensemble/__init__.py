"""Supervised ML ensemble model plugins.

Explicitly wired (not auto-registered): `default_models()` below is the one
place to touch when adding, removing, or swapping a model.

Deferred to later modules: CatBoost, LightGBM (see `xgboost_classifier.py`),
and SHAP for the two ensemble combiners (see `stacking_ensemble.py`/
`voting_ensemble.py`). Model persistence lives in
`application/model_persistence/`.
"""

from __future__ import annotations

from pdf_forensics.application.ml_ensemble.ports import SupervisedModelPlugin
from pdf_forensics.plugins.ml_ensemble.extra_trees_classifier import ExtraTreesModel
from pdf_forensics.plugins.ml_ensemble.logistic_regression_classifier import (
    LogisticRegressionModel,
)
from pdf_forensics.plugins.ml_ensemble.random_forest_classifier import RandomForestModel
from pdf_forensics.plugins.ml_ensemble.stacking_ensemble import StackingEnsembleModel
from pdf_forensics.plugins.ml_ensemble.voting_ensemble import VotingEnsembleModel
from pdf_forensics.plugins.ml_ensemble.xgboost_classifier import XGBoostModel


def default_models() -> tuple[SupervisedModelPlugin, ...]:
    return (
        RandomForestModel(),
        ExtraTreesModel(),
        LogisticRegressionModel(),
        XGBoostModel(),
        StackingEnsembleModel(),
        VotingEnsembleModel(),
    )


__all__ = ["default_models"]
