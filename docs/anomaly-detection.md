# Module 5: Anomaly Detection

Covers `src/pdf_forensics/domain/anomaly_detection/`,
`src/pdf_forensics/application/anomaly_detection/`, and
`src/pdf_forensics/plugins/anomaly_detection/`.

## What this is, and isn't

Four unsupervised detectors — Isolation Forest, One-Class SVM, Local Outlier
Factor (novelty mode), and a small PyTorch autoencoder — each score how
unusual a document's numeric feature vector is relative to a reference batch
they were fit on. Per the platform's brief, this is **independent of fraud
labels**: fitting only ever sees `FeatureSet`s, never a ground-truth label.
Not a risk score (a later module combines this with Rule Engine findings and
ML Ensemble output); one signal source among several.

## The first module with real dependencies

Modules 1-4 were deliberately stdlib-only. Real anomaly detection needs real
ML libraries — confirmed with the user before adding them:

- `scikit-learn` for Isolation Forest, One-Class SVM, and LOF.
- `torch`, from the CPU-only wheel index (`pyproject.toml`'s
  `[[tool.poetry.source]]` for `pytorch-cpu`) — this machine is CPU-only, so
  the default CUDA-bundled wheel would be a multi-GB dependency for nothing.

## Detectors are stateful (unlike Modules 2 and 4's plugins)

`AnomalyDetectorPlugin` (`application/anomaly_detection/ports.py`) has both
`fit()` and `score()` — an ML model has to be trained before it can predict
anything. Calling `score()` before `fit()` raises `RuntimeError` uniformly
across all four detectors (the three sklearn-based ones don't rely on
sklearn's own `NotFittedError` — an explicit check keeps the failure mode
identical everywhere).

## Feature vectorization

Only `INTEGER`/`FLOAT`/`BOOLEAN`-typed features go into a detector's input
vector (`application/anomaly_detection/feature_vectorizer.py`) — histograms
(`DICT`), producer strings (`STRING`), etc. are out of scope this iteration.
A numeric feature whose value is `None` (e.g. `catalog.page_count` when
`/Pages` doesn't resolve) is *omitted*, not coerced to `0.0` — "unknown" and
"zero" would otherwise look identical to every detector.

Since different documents surface different *sets* of feature names (e.g.
`metadata.*` has 14 keys with `/Info` present, 1 without), each detector
fits its own `sklearn.feature_extraction.DictVectorizer` on the training
batch's vocabulary; a document at score-time missing a training-time key
gets `0.0` there (DictVectorizer's standard behavior).

## Sign convention

`AnomalyScore.score`: higher always means more anomalous, across all four
detectors, even though the underlying algorithms don't agree on that by
default. The three sklearn estimators share a `decision_function` convention
(positive = inlier, negative = outlier), so each reports `-decision_function`;
the autoencoder reports its reconstruction MSE directly, which already
increases with anomalousness. `is_anomaly`: sklearn detectors use their own
`decision_function(x) < 0` boundary; the autoencoder compares against
`mean + 3·std` of the *training batch's own* reconstruction errors.

## Explicitly deferred

Persisting fitted detectors to disk (a future Model Registry concern) and
combining multiple detectors' scores into one signal (the future Risk Report
module). This module produces raw per-detector scores only.

## Usage

```python
from pdf_forensics.application.anomaly_detection.fit_anomaly_detectors_use_case import (
    FitAnomalyDetectorsUseCase,
)
from pdf_forensics.application.anomaly_detection.detect_anomalies_use_case import (
    DetectAnomaliesUseCase,
)
from pdf_forensics.plugins.anomaly_detection import default_detectors

detectors = default_detectors()
FitAnomalyDetectorsUseCase(detectors).execute(training_feature_sets)  # e.g. from
# application/training_data/build_training_dataset_use_case.py's TrainingDataset

report = DetectAnomaliesUseCase(detectors).execute(new_document_feature_set)
for score in report:
    print(score.detector_id, score.score, score.is_anomaly)
```
