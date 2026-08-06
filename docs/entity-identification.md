# Entity Identification

Covers `src/pdf_forensics/domain/entity_identification/`,
`src/pdf_forensics/application/entity_identification/`, and
`src/pdf_forensics/plugins/entity_identification/`.

## What this is

A multiclass classifier that predicts which known real-world entity most
likely produced a document — the platform brief's `Generator Confidence`,
scoped to what's actually buildable today (see "What this is NOT" below).
Feeds two things: a new `entity_consistency` component in Module 8's Risk
Report (confidence-based), and a new, specific `entity_template_mismatch`
Rule Engine rule (`docs/rule-engine.md`) that checks a confidently-predicted
entity against a known structural invariant for that entity.

## Why this became feasible

Comparing real documents from 3 Argentine entities (ANSES, Municipalidad de
La Rioja, Municipalidad de San Salvador de Jujuy) surfaced a strong,
consistent pattern: each entity's PDFs share the same `/Producer` string, and
in ANSES's case, an *identical* `structure_hash`/`page_tree_hash` (Module
3's fingerprint) across documents belonging to four different people:

| Entity | Producer | Notes |
|---|---|---|
| ANSES | `iTextSharp™ 5.5.13.4 ©2000-2024 iText Group NV (AGPL-version); modified using iText® Core 7.2.3...` | Identical structure/page-tree hash across all 4 samples seen |
| Municipalidad de San Salvador de Jujuy | `iTextSharp™ 5.5.8 ©2000-2015 iText Group NV (AGPL-version)` | Older iTextSharp version than ANSES's |
| Municipalidad de La Rioja (Capital) | `mPDF 5.7` | A different PDF library entirely — trivially separable from the other two |

This is exactly the kind of pattern a classifier can learn cheaply and
reliably, even from a handful of real documents per entity.

## A second invariant, found by comparing real documents across entities

`catalog.has_acroform` (Module 2) is a perfect per-entity invariant across
every real document sampled so far:

| Entity | `catalog.has_acroform` | Notes |
|---|---|---|
| ANSES | always `True` | Matches the "Firmado Digitalmente por ANSES" badge visible on genuine receipts — a real `/AcroForm` digital-signature structure, not just text |
| Municipalidad de La Rioja | always `False` | |
| Municipalidad de San Salvador de Jujuy | always `False` | |

This is specific and verifiable enough to be its own Rule Engine rule
(`entity_template_mismatch`, `docs/rule-engine.md`) rather than folded into
the confidence-only `entity_consistency` score — a document identified as
ANSES without an AcroForm is a concrete, named contradiction, not just "low
confidence."

Numeric structural features (`general.object_count`, `xref.in_use_entry_count`)
did **not** turn out to be clean per-entity invariants yet — they split into
two disjoint clusters per entity: the handful of untouched real documents,
and the synthetic field-substituted variants (`pdf-forensics-benchmark`'s
`FieldSubstitutionGenerator`), which get a different low-level byte
structure from PyMuPDF's resave (a known, documented limitation of that
generator). More untouched real documents per entity, not more synthetic
variants, is what would make these usable as "expected envelope" checks.

## What this is NOT

The brief's `Generator Confidence` implies comparing what a document
*claims* to be (its visible institution branding/text) against what it
*structurally* looks like — catching a document that says "ANSES" but was
never produced by ANSES's actual system. This platform can't do that yet:
Module 2's feature extraction is structural/metadata only (xref, objects,
`/Info` dict) — there's no page-text-layout extraction to read what a
document's content actually displays.

What's implemented instead: "does this document's structure/producer
confidently match one of the real-world templates the classifier has
actually been trained on." A real, useful, narrower signal — not a fake
stand-in for the full check.

## Domain (`domain/entity_identification/`)

- **`entity_prediction.py`**: `EntityPrediction` (frozen dataclass) —
  `classifier_id`, `predicted_entity`, `confidence`, `probabilities` (every
  known class, not just the winner).
- **`entity_identification_report.py`**: `EntityIdentificationReport` — one
  `EntityPrediction` per configured classifier, same shape as Module 6's
  `MlEnsembleReport`.

## Application (`application/entity_identification/`)

- **`feature_vectorizer.py`**: `select_entity_features` — every numeric
  feature `ml_shared.select_numeric_features` selects, plus the raw
  `metadata.producer`/`metadata.creator` strings (the strongest signal
  observed). `DictVectorizer` one-hot-encodes strings automatically.
- **`ports.py`**: `EntityClassifierPlugin` Protocol — same stateful shape as
  Module 6's `SupervisedModelPlugin` (`fit()` before `predict()`).
- **`train_entity_classifier_use_case.py`** / **`identify_entity_use_case.py`**:
  mirror Module 6's `TrainModelsUseCase`/`PredictUseCase`. Training accepts
  the same optional `sample_weight` as Modules 5/6
  (`application/ml_shared/sample_weighting.py`).

## Plugins (`plugins/entity_identification/`)

Deliberately **one** classifier (`RandomForestEntityClassifier`), not an
ensemble like Module 6's six models — the training corpus (a handful of real
documents per entity) doesn't justify more; `default_entity_classifiers()`
is the one place to add more later.

## Training data

`scripts/_entity_labels.py` maps a benchmark sample's `generator` string
(`external:<filename-stem>`) to a known entity via substring patterns —
extended there whenever a new real document is added for an entity, or a new
entity's documents show up. Every sample derived from a given real document
(its transformations, and any synthetic field-substituted variants) inherits
that document's entity label via `original_id`.

## Usage

```python
from pdf_forensics.application.entity_identification.identify_entity_use_case import (
    IdentifyEntityUseCase,
)

entity_report = IdentifyEntityUseCase(entity_classifiers).execute(feature_set)
for prediction in entity_report:
    print(f"{prediction.predicted_entity} ({prediction.confidence:.0%})")
```
