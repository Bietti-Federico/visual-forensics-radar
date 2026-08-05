# Module 2: Feature Extraction

Covers `src/pdf_forensics/domain/features/`, `src/pdf_forensics/application/feature_extraction/`,
and `src/pdf_forensics/plugins/features/` — turning an already-parsed `PdfDocument`
(Module 1) into a `FeatureSet` of typed, documented forensic features.

## Scope: honest, not padded

Following the same principle Module 1 used (and the sibling `pdf-forensics-benchmark`
project's own feature extractor): only features computable *today* from the
existing `PdfDocument` model are implemented. No font/image/`/Resources`
tree-walking, no XMP semantic parsing, no signature validation, no
linearization detection, no page-tree walk beyond `/Pages/Count`. Ten
categories, ~45 features — real, none placeholders.

## Why a `Feature` carries provenance, not just a value

Every `Feature` records `source` (the exact field/method it came from) and
`confidence` (always `1.0` this iteration — every feature is a direct
structural read; the field exists for later heuristic/derived features) so a
report or feature store never needs to go back to the extractor's source code
to answer "what does this column mean."

## Plugin architecture: explicit wiring, not auto-registration

`FeatureExtractorPlugin` (`application/feature_extraction/ports.py`) is a
`Protocol`, not an ABC — an extractor needs no base-class import to qualify.
It's a *port*: the application layer's use case depends on it, and the outer
`plugins/features/` package depends inward on it, never the reverse.

This deliberately does **not** reuse the sibling `pdf-forensics-benchmark`
project's `__init_subclass__` auto-registration convention: that pattern only
fires if the defining module is actually imported (forcing either an eager
`import *` or fragile import-order reliance), and shared registry state
complicates isolated per-extractor unit tests. Instead,
`plugins/features/__init__.py`'s `default_feature_extractors()` is an
explicit tuple — the one place to touch when adding, removing, or swapping an
extractor.

`FeatureExtractionUseCase` takes its extractors as a required constructor
argument rather than defaulting to `default_feature_extractors()` itself —
that would make the application layer import the outer `plugins/` package,
inverting the dependency rule. The composition root (tests today, a future
CLI/API) wires concrete extractors in.

## Categories and where each comes from

| Category | Source on `PdfDocument` |
|---|---|
| `general` | `pdf_version`, `raw_size`, `len(objects)`, `len(revisions)` |
| `metadata` | `/Info` dict via `trailer.get_ref("Info")` → `resolve()` |
| `trailer` | `trailer.entries` / `.get_ref` / `.get` |
| `catalog` | `/Root` → resolved dict; `/Pages/Count` (no tree walk) |
| `xref` | `latest_revision.xref_entries`, `.is_xref_stream`, `/XRefStm` |
| `incremental_updates` | `len(revisions)` |
| `objects` | `isinstance` histogram over `objects.values()`; `DUPLICATE_OBJECT_ID` anomalies |
| `streams` | `PdfStream.filter_names`, `raw_data` length; `UNSUPPORTED_FILTER` anomalies |
| `security` | `is_encrypted`, raw `encryption_dict` (no decryption) |
| `statistics` | `anomalies` grouped by severity/code |

## `pdf_version`: a small, justified Module 1 extension

`PdfDocument` gained a `pdf_version: str | None` field, captured in
`PdfDocumentParser.parse()` where it already scans for the `%PDF-` header for
the `NotAPdfError` check — near-zero marginal cost, and it keeps this module
a pure post-parse consumer of `PdfDocument` rather than re-touching raw bytes.
