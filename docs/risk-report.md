# Module 8: Risk Report (final module of this pass)

Covers `src/pdf_forensics/domain/risk/` and
`src/pdf_forensics/application/risk_report/`.

## What this is

The final deliverable of the pipeline: a weighted **0-100 risk score**,
packaged together with Module 7's `ExplanationReport` — matching the
platform brief's own illustrative example, which shows the score and its
reasons/SHAP top features together as one output.

## All seven of the brief's inputs are now represented

The brief lists: Rule Engine, **Generator Confidence**, Anomaly Score, ML
Probability, **Fingerprint Similarity**, Structural Confidence, Metadata
Confidence.

- **Generator Confidence** is covered by `entity_consistency`, backed by the
  Entity Identification module (`application/entity_identification/`) —
  see below.
- **Fingerprint Similarity** is covered *in spirit, not literally* by
  `signature_integrity`, backed by the Signature Verification module
  (`application/signature_verification/`, `docs/signature-verification.md`)
  — Module 3 still has no reference corpus of known-good fingerprints to
  compare against, but a cryptographically verified embedded signature is a
  much stronger per-document integrity signal than a similarity search
  would have been anyway.

## `entity_consistency` (Entity Identification)

Trains a multiclass classifier (currently one `RandomForestEntityClassifier`,
see `plugins/entity_identification/`) on real documents grouped by which
known entity issued them, then scores `1 - confidence` for a new document:
low confidence in matching ANY known entity's structure/producer pattern is
the risk signal.

This is **not** the full "claimed vs. actual issuer" check the brief's
`Generator Confidence` ultimately implies — this platform has no page-text
extraction yet (Module 2 is structural/metadata features only), so there's
no way to read which institution a document's own *visible content* claims
to be from and compare it against the structural prediction. What's
implemented is a genuine, narrower signal: "does this document's structure
confidently resemble one of the real-world templates we've actually seen."
See `docs/entity-identification.md`.

## `signature_integrity` (Signature Verification)

Validates any embedded PKCS#7 digital signature: does its cryptographic
digest still match the signed bytes, is the signature itself valid, and
does it cover the whole file. See `docs/signature-verification.md` for the
full scoring table and why trust-chain validation is deliberately excluded.

## The weights are a documented starting point, not a calibrated model

There's no labeled dataset large enough yet to empirically fit these weights
— that's exactly what Training Data Ingestion → ML Ensemble is for, going
forward. `RiskWeights` is a constructor-injectable dataclass (defaults sum to
1.0, validated) specifically so recalibration never requires touching this
module's structure.

## The seven components

| Component | Formula | Why |
|---|---|---|
| `rule_engine` | Noisy-OR over `RuleFinding` severities (`CRITICAL=0.9, WARNING=0.5, INFO=0.15`) | Independent evidence saturates toward 1.0 without a plain sum overshooting it |
| `ml_probability` | Mean of `ModelPrediction.probability` | Probabilities *are* comparable across Module 6's models by design (all are P(manipulated)) |
| `anomaly_detection` | Noisy-OR over `is_anomaly` flags only (weight 0.7 each) | Raw `.score` isn't comparable across Module 5's detectors — only the boolean flag is |
| `structural` | `min(1, anomaly_count_total / object_count * 10)` | A general anomaly-density signal, distinct from the specific patterns Rule Engine checks |
| `metadata` | `1.0` if no `/Info` dict; else fraction of 6 metadata fields missing | Missing metadata correlates with scrubbing, same rationale as Module 4's `missing_info_dictionary` rule |
| `entity_consistency` | `1 - mean(EntityPrediction.confidence)` | Low confidence in matching any known entity's structural template is itself a signal, narrower than a true claimed-vs-actual check (see above) |
| `signature_integrity` | Worst-finding-wins across embedded signatures (`0.0` none/valid, `0.7` partial coverage, `0.9` invalid, `1.0` broken digest) | See `docs/signature-verification.md` |

Final score: `round(clamp(Σ(score·weight), 0, 1) * 100)`.

## Usage

```python
from pdf_forensics.application.risk_report.generate_risk_report_use_case import (
    GenerateRiskReportUseCase,
)

risk_report = GenerateRiskReportUseCase().execute(
    feature_set, rule_report, anomaly_report, ml_report, shap_explanations,
    entity_report, signature_report,
)
print(f"Risk Score: {risk_report.risk_score}")
for component in risk_report.components:
    print(f"  {component.name}: {component.score:.2f} (weight {component.weight})")
for reason in risk_report.explanation.reasons:
    print("-", reason)
```
