# Module 6: ML Ensemble

Covers `src/pdf_forensics/domain/ml_ensemble/`, `src/pdf_forensics/application/ml_ensemble/`,
`src/pdf_forensics/plugins/ml_ensemble/`, and the shared
`src/pdf_forensics/application/ml_shared/feature_vectorizer.py`.

## What this is, and isn't

The **supervised** counterpart to Module 5's unsupervised anomaly detection:
six classifiers trained on labeled data (genuine vs. transformed/manipulated,
from the Training Data Ingestion module) that each predict a probability and
(for four of the six) explain that specific prediction via SHAP. Not the
Risk Report (a later module combines this with Rule Engine findings and
Anomaly Detection scores into one number) and not Explainability (which
turns SHAP values into human-readable text — this module only produces the
raw numbers).

## Scope: one gradient-boosting library, not three

The spec asks for CatBoost, LightGBM, and XGBoost. Confirmed with the user:
only **XGBoost** is implemented — the three overlap functionally, and each
is a heavy native-build dependency. CatBoost/LightGBM are deferred, not faked.

## Target label and score convention

Binary target: `is_original` from Training Data Ingestion's
`BenchmarkSampleRef`. The positive class is `False` (manipulated) —
`ModelPrediction.probability` is **P(manipulated)**, so "higher = more
suspicious" holds consistently with Module 5's "higher = more anomalous."

## Shared refactor: feature vectorization

`select_numeric_features` moved from `application/anomaly_detection/` to
`application/ml_shared/` — nothing about it was anomaly-detection-specific,
and this module needs the exact same numeric-feature selection for
supervised training. One implementation, not two that could drift (same
rationale as Module 3's `cos_type_name` extraction from Module 2).

## The six models

Four base classifiers with real SHAP support, sharing
`_sklearn_classifier_base.py` (`DictVectorizer` fit/transform + fit/predict/
explain plumbing, mirroring Module 5's `_sklearn_base.py`):

| Model | Estimator | Explainer |
|---|---|---|
| `random_forest` | `sklearn.ensemble.RandomForestClassifier` | `shap.TreeExplainer` |
| `extra_trees` | `sklearn.ensemble.ExtraTreesClassifier` | `shap.TreeExplainer` |
| `logistic_regression` | `sklearn.linear_model.LogisticRegression` | `shap.LinearExplainer` |
| `xgboost` | `xgboost.XGBClassifier` | `shap.TreeExplainer` |

Two ensembles over those same four estimators, predictions only:

| Model | Combiner |
|---|---|
| `stacking_ensemble` | `sklearn.ensemble.StackingClassifier` (meta-learner: `LogisticRegression`) |
| `voting_ensemble` | `sklearn.ensemble.VotingClassifier` (soft voting) |

## SHAP for the two ensembles is explicitly deferred

A heterogeneous stacked/voted combiner needs `shap.KernelExplainer`
(model-agnostic, but slow and needs careful background-sampling tuning to be
reliable) — not implemented this iteration. `_build_explainer()` returns
`None` for these two, and `explain()` raises `NotImplementedError` — distinct
from the `RuntimeError` raised when a model hasn't been fit yet at all.
`ExplainPredictionUseCase` treats `NotImplementedError` as "no explanation
available," not a failure, and skips that model rather than erroring the
whole batch.

## Explicitly deferred

CatBoost, LightGBM, model persistence (a future Model Registry concern, same
as Module 5), SHAP for the two ensembles.

## Usage

```python
from pdf_forensics.application.ml_ensemble.train_models_use_case import TrainModelsUseCase
from pdf_forensics.application.ml_ensemble.predict_use_case import PredictUseCase
from pdf_forensics.application.ml_ensemble.explain_prediction_use_case import (
    ExplainPredictionUseCase,
)
from pdf_forensics.plugins.ml_ensemble import default_models

models = default_models()
labels = [sample.source.is_original for sample in training_dataset.samples]
feature_sets = [sample.features for sample in training_dataset.samples]
TrainModelsUseCase(models).execute(feature_sets, labels)

report = PredictUseCase(models).execute(new_document_feature_set)
for prediction in report:
    print(prediction.model_id, prediction.probability, prediction.predicted_label)

for explanation in ExplainPredictionUseCase(models).execute(new_document_feature_set):
    print(explanation.model_id, dict(zip(explanation.feature_names, explanation.shap_values)))
```
