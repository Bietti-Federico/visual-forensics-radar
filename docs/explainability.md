# Module 7: Explainability

Covers `src/pdf_forensics/domain/explainability/` and
`src/pdf_forensics/application/explainability/`.

## What this is, and isn't

A pure aggregator: takes Module 4's `RuleEvaluationReport`, Module 5's
`AnomalyDetectionReport`, Module 6's `MlEnsembleReport` and
`list[ShapExplanation]`, and produces an `ExplanationReport` — plain-language
"reasons" plus a per-model "top features" list. Not a risk score; that's the
next (final) Risk Report module, which will combine this module's reasons
with a weighted numeric score.

## No new dependencies, no fabricated reasons

This module has zero new dependencies — it's pure domain/application logic
over types that already exist. Every reason string traces back to something
Modules 4-6 actually computed:

- Every `RuleFinding.explanation`, sorted CRITICAL → WARNING → INFO.
- One reason per `AnomalyScore` where `is_anomaly` is `True`.
- One reason per `ModelPrediction` where `predicted_label` is `True`.

No invented example text like the platform brief's illustrative "Producer
inconsistent" or "Ghostscript rewrite probability" — those were only
illustrative in the brief, not something this platform fabricates.

## Top features stay per-model

`TreeExplainer` and `LinearExplainer` operate in different output spaces
(margin/log-odds vs. linear score) — merging their raw `shap_value`
magnitudes into one ranked list would imply a comparability that doesn't
exist. Each `ShapExplanation`'s own top-N (default 5, by `abs(shap_value)`)
is kept, tagged with its `model_id`.

## Usage

```python
from pdf_forensics.application.explainability.generate_explanation_use_case import (
    GenerateExplanationUseCase,
)

explanation = GenerateExplanationUseCase().execute(
    rule_report, anomaly_report, ml_report, shap_explanations
)
for reason in explanation.reasons:
    print("-", reason)
for feature in explanation.top_features:
    print(feature.model_id, feature.feature_name, feature.shap_value)
```
