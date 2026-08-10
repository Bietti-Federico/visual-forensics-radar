# Verification API + Training-Data Frontend

Covers `src/pdf_forensics_api/`,
`src/pdf_forensics/application/document_scoring/`,
`src/pdf_forensics/application/model_training/`,
`src/pdf_forensics/application/training_corpus/`,
`src/pdf_forensics/infrastructure/training_corpus/`, and the per-entity
redesign of `src/pdf_forensics/application/model_persistence/`.

## Why this exists

Modules 1-9 were exercised only through one-off CLI scripts run manually.
This is the production integration point: a document-upload workflow calls
`POST /verify` to score a file, and a minimal internal frontend lets someone
grow the training corpus over time — including for entities the platform
has never seen — without touching code.

Three constraints shaped this beyond "wrap the pipeline in an API":

1. **Verification must never write to disk.** No per-request logs, no
   temp files. `/verify` processes the uploaded bytes entirely in memory;
   the JSON response is the only output.
2. **Disk usage only grows when someone deliberately adds a training
   file.** No fabricated/synthetic data, no version history for the model
   file — one `model_store.joblib`, overwritten in place on retrain.
3. **Retraining is a manual, explicit action.** Uploading a training file
   never triggers a retrain by itself — it just adds a file under
   `training_corpus/`. A separate `POST /retrain` reads the whole corpus
   and refits everything.

## Why per-entity, not global, model fitting

Fitting Anomaly Detection (Module 5) globally across every entity pooled
together was tried first and is actively misleading: with only the real
documents on hand, it flagged 6 of 10 genuine in-sample documents as
anomalous, because ANSES/La Rioja/Jujuy have structurally different
templates and pooling them makes each look "anomalous" relative to the
others. Fitting **per entity** — an ANSES document is only ever compared
against other ANSES documents — removes nearly all of that noise.

ML Ensemble (Module 6) has the same problem in a sharper form: it needs
labeled *manipulated* examples to learn from, and this platform has very
few confirmed-fraud cases. Training it on fabricated field-substitution
variants doesn't buy real signal — it teaches the model to recognize the
fabrication technique, not fraud. So the production retrain flow never
touches synthetic data (that stays in the sibling `pdf-forensics-benchmark`
project for R&D only) and instead **gates ML Ensemble activation
automatically, per entity**, on a minimum count of real genuine and
confirmed-fraud examples for that entity:

```python
# application/model_training/retrain_models_use_case.py
ANOMALY_MIN_GENUINE_PER_ENTITY = 2
ML_ENSEMBLE_MIN_GENUINE_PER_ENTITY = 5
ML_ENSEMBLE_MIN_CONFIRMED_FRAUD_PER_ENTITY = 3
```

An entity below the ML Ensemble thresholds is scored with
`RiskWeights.without_ml_probability()` — `ml_probability`'s weight is
zeroed and the remaining six components are rescaled to still sum to 1.0
— rather than shipping a classifier trained on too little data. Once an
entity crosses both thresholds at a later retrain, it automatically starts
using the full `RiskWeights()` on the next `/verify` call. No code change,
no manual flag flip.

## Training corpus: plain directories, entity = folder name

```
training_corpus/
  genuine/
    ANSES/*.pdf
    LA_RIOJA/*.pdf
    <NEW_ENTITY>/*.pdf        # adding an entity = a new folder, zero code changes
  confirmed_fraud/
    ANSES/AMPF-095000-16859.pdf
    ...
```

No manifest, no CSV. `infrastructure/training_corpus/
filesystem_training_corpus_reader.py` scans both trees; the folder name
under `genuine/` or `confirmed_fraud/` *is* the entity label. This
replaces the CSV-manifest approach in `docs/training-data-ingestion.md`,
which stays as-is for R&D work against the benchmark project's synthetic
output — the two flows are independent and read different directories.

Adding a folder needs **zero** code changes for Entity Identification,
Anomaly Detection, ML Ensemble, or the template-invariant check below —
every one of them is fitted per entity from whatever's on disk. The one
thing that stays a manual, hand-authored step for a genuinely new kind of
check is adding a brand-new *rule* to `default_rules()`
(`plugins/rules/__init__.py`) — a structural assertion nobody has
expressed as a feature yet. Per-entity invariants over existing features
are not that case; see the next section.

## Auto-learned per-entity template invariants

The `entity_template_mismatch` finding used to be a hand-written,
hand-maintained dict of "this entity's genuine documents always have
X" facts (`_ENTITY_TEMPLATE_INVARIANTS`) — a developer had to inspect a new
entity's documents and edit source code before it contributed anything for
that entity. It's now mined automatically at retrain time, per entity, by
`application/entity_invariants/fit_entity_invariants_use_case.py`: any
boolean feature, or any key of a dict-valued histogram feature (e.g.
`streams.filter_histogram`), whose value is identical across every genuine
document of that entity becomes a `LearnedInvariant`. Continuous/scalar
features (byte counts, object counts, ...) are deliberately never mined —
see `docs/entity-identification.md` for why that would just produce
constant false positives once the corpus grows.

Gated the same way as the other per-entity models, by its own minimum
sample count:

```python
TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY = 4
```

Below that, an entity has no learned invariants yet (`()`), same shape as
"not enough data yet" for Anomaly Detection/ML Ensemble. At verify time,
`CheckEntityInvariantsUseCase` compares the document against the predicted
entity's learned invariants and produces the same `RuleFinding` shape (same
`rule_id`, `entity_template_mismatch`) the old hand-written rule did, with
an explanation generated from the actual/expected values instead of
hand-written prose. `/retrain`'s response and the frontend surface
`invariant_count` per entity, the same way they already surface
`anomaly_detection_fitted`.

## Model persistence: schema v3.0.0

`SaveTrainedModelsUseCase`/`LoadTrainedModelsUseCase` changed shape from a
flat global bundle to per-entity, then gained the invariants field above:

```python
{
  "schema_version": "3.0.0",
  "entity_classifiers": {...},   # one global classifier, unchanged (Module 9)
  "per_entity": {
    "ANSES": {
      "detectors": {...} | (),   # () if genuine_count < ANOMALY_MIN_GENUINE_PER_ENTITY
      "models": {...} | (),      # () if not ml_ensemble_ready
      "invariants": (...) | (),  # () if genuine_count < TEMPLATE_INVARIANT_MIN_GENUINE_PER_ENTITY
      "genuine_count": 4,
      "confirmed_fraud_count": 1,
      "ml_ensemble_ready": False,
    },
    ...
  },
}
```

Each schema bump (`1.0.0` → `2.0.0`: per-entity bundles; `2.0.0` → `3.0.0`:
added `invariants`) is a breaking change with no migration path for old
bundles; retrain from the corpus to produce a current one.

## `ScoreDocumentUseCase`: one shared scoring path

`application/document_scoring/score_document_use_case.py` is what both
`/verify` and `scripts/score_document.py` call — extracted so the API and
any future CLI can't drift apart the way the old scripts (which each
duplicated the Module 1-9 orchestration by hand) did. Given `pdf_bytes`
and an already-loaded `(entity_classifiers, per_entity)` bundle:

1. Parse + extract features + fingerprint (Modules 1-3).
2. `IdentifyEntityUseCase` → top-predicted entity.
3. Look up that entity's `detectors`/`models`/`invariants`/`ml_ensemble_ready`
   from `per_entity`. An unrecognized entity (no bundle, e.g. nothing
   trained yet) scores with empty Anomaly Detection, ML Ensemble, and
   invariant-check results — the same as "nothing flagged," not an error.
4. Rule Engine over the plain `FeatureSet`, plus
   `CheckEntityInvariantsUseCase` against that entity's learned invariants
   (see below), plus Signature Verification (Module 10).
5. Picks `RiskWeights()` or `RiskWeights().without_ml_probability()`
   per the entity's `ml_ensemble_ready` flag.

Entirely a pure function of its inputs: no disk I/O, no mutation of the
bundle — safe to call from a request handler with the bundle held in
memory for the life of the process.

## API endpoints

`src/pdf_forensics_api/app.py`, a FastAPI app kept as a separate top-level
package from `pdf_forensics` — the core library stays free of web-framework
concerns, the same reasoning `scripts/` is a thin caller of the library
rather than part of it.

| Endpoint | Effect |
|---|---|
| `GET /` | Serves the static frontend (`static/index.html`). |
| `POST /verify` | Multipart file upload. Runs `ScoreDocumentUseCase` entirely in memory; nothing is written to disk. Returns the risk score, its components, entity prediction, signature results, and explanation reasons as JSON. `422` if the file isn't a readable PDF. |
| `POST /training-data/genuine` | Multipart file + `entity` form field. Validates the file is a readable PDF (`422` otherwise), then saves it under `training_corpus/genuine/<entity>/`. Does **not** retrain. |
| `POST /training-data/confirmed-fraud` | Same as above, under `training_corpus/confirmed_fraud/<entity>/`. |
| `GET /training-data/summary` | Live directory scan (not cached) — per-entity genuine/confirmed-fraud counts, so the frontend can show what's changed since the last retrain. |
| `POST /retrain` | Runs `RetrainModelsUseCase` against the whole corpus, overwrites `model_store.joblib`, then reloads the in-memory bundle so `/verify` reflects it immediately — no process restart needed. Returns per-entity counts and readiness flags. |

The trained bundle loads once at process startup via FastAPI's `lifespan`
context manager and lives in `app.state`; `/retrain` is the only thing
that reloads it.

**Path safety**: both `entity` and the uploaded filename become filesystem
path segments under `training_corpus/`, so both are sanitized
(`_sanitize_path_component`) against path traversal before being used —
`entity="../../etc"` or a crafted filename can't escape the corpus
directory.

**Upload size**: capped at 25 MB (`_MAX_UPLOAD_BYTES`) — this is a document
verification API, not a general file store.

### Configuration

Both paths are read fresh from the environment on every call (not cached
at import time), so tests can point them at a `tmp_path` before the app
starts:

| Env var | Default |
|---|---|
| `PDF_FORENSICS_TRAINING_CORPUS_DIR` | `training_corpus` |
| `PDF_FORENSICS_MODEL_STORE_PATH` | `model_store.joblib` |

### Running it

```bash
poetry run uvicorn pdf_forensics_api.app:app --host 0.0.0.0 --port 8000
```

## Frontend

`src/pdf_forensics_api/static/index.html` — a single static file, vanilla
JS `fetch()`, no build step, served directly by the same backend at `/`.
Three sections: drag-and-drop verify (renders the JSON result readably),
an upload form for genuine/confirmed-fraud training files (entity is a
free-text field, not a fixed dropdown, since adding a new entity is meant
to need no code change), and a retrain button that shows the live corpus
summary and the last retrain result.

## A signature-verification pitfall this surfaced

Manually running `/verify` end-to-end (not just the unit test suite)
against a real, genuinely signed ANSES document turned up a bug that no
unit test had caught: signatures that verified correctly when called from
a plain script came back empty (`"signatures": []`) through the API.

The cause: pyHanko's `validate_pdf_signature` wraps an async
implementation in its own internal `asyncio.run(...)`, which raises
`RuntimeError: asyncio.run() cannot be called from a running event loop`
when called from within FastAPI's already-running event loop (any `async
def` route handler runs on one). `pyhanko_adapter.py`'s per-signature
`try/except Exception: continue` — there specifically so one malformed
signature doesn't take down the whole report — silently absorbed that
`RuntimeError` too, dropping every signature found on documents verified
through the API while leaving CLI usage unaffected.

Fixed by calling pyHanko's own async entry point
(`async_validate_pdf_signature`) directly and running it through a small
`_run_coro_sync` helper that detects whether a loop is already running: if
not, `asyncio.run()` as before; if so, the coroutine runs on a dedicated
thread so it never collides with the caller's loop. Regression-tested in
`tests/application/signature_verification/test_verify_signatures_use_case.py`
by calling `VerifySignaturesUseCase` from inside `asyncio.run(...)` and
asserting the signature is still found — the exact condition that dropped
it before the fix.

## Testing

- `RetrainModelsUseCase`/`BuildTrainingCorpusUseCase`: `tmp_path`-based
  fake corpora — per-entity detector fitting skipped below threshold, ML
  Ensemble readiness flipping exactly at the threshold boundary, a
  brand-new entity folder picked up with no code changes.
- `ScoreDocumentUseCase`: confirms the right `RiskWeights` variant is
  picked for an ML-ready vs. not-yet-ready entity, and that an
  unrecognized entity still scores (with the reduced weight set).
- API layer: `fastapi.testclient.TestClient` against every endpoint,
  each test pointed at a `tmp_path`-scoped corpus/model path via
  `monkeypatch.setenv` (never the real `training_corpus/`) — upload,
  confirm the summary reflects it, retrain, confirm `/verify` uses the
  new model.
- Manually verified against the real corpus in Docker: uploaded all 10
  real genuine documents (ANSES/La Rioja/Jujuy) and the one confirmed-fraud
  case (AMPF) through the actual HTTP endpoints, retrained, then verified
  both the fraud case and a genuine document — confirming the same
  findings (producer mismatch, elevated risk score on the fraud case; low
  risk score and a correctly-found intact signature on the genuine one)
  seen in earlier script-based runs, with the honest caveat that entity
  confidence is lower on this much smaller real-only corpus than on the
  earlier synthetic-augmented experiments.
