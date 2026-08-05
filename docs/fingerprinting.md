# Module 3: Fingerprint Generation

Covers `src/pdf_forensics/domain/fingerprint/` and
`src/pdf_forensics/application/fingerprinting/` — turning a `PdfDocument`
(Module 1) plus its `FeatureSet` (Module 2) into a comparable, hash-based
`PdfFingerprint`. Not generator identification (a separate future ML module)
and not risk scoring (later modules) — just deterministic SHA256 hashes plus
two raw metadata strings.

## No new byte-level parsing

This module only traverses the already-built `PdfDocument` graph (dict/
reference lookups, the same operations Module 2's extractors perform). It
never re-tokenizes raw bytes.

## Why `generator`/`producer`/`pdf_version` don't come from the `FeatureSet`

It would be tempting to read `feature_set.by_name("metadata.producer")` since
Module 2 already extracted it. Deliberately not done: that couples this
module to Module 2's feature-name strings by convention only, with nothing
(compiler or test) to catch a rename. Instead this module re-resolves
`/Info` directly from `PdfDocument` — a small, intentionally-duplicated
~10-line helper — and reads `document.pdf_version` directly. `feature_set` is
used for exactly one field, `feature_hash`, since that field's entire meaning
is "hash of whatever Module 2 actually extracted."

## Hash fields and what makes each one comparable

| Field | Computed from | Comparable across... |
|---|---|---|
| `xref_hash` | Latest revision's xref entries, sorted | Same file, byte-identical saves |
| `structure_hash` | Every object's `(obj_num, generation, cos_type_name)`, sorted | Structurally-identical documents saved at different times/offsets |
| `metadata_hash` | Every `METADATA`-category feature, sorted | Documents with identical `/Info` content |
| `page_tree_hash` | `/Root/Pages`'s `Type`/`Count`/ordered `Kids` refs | Documents with the same page-tree shape |
| `feature_hash` | Canonical JSON of the full `FeatureSet` | Documents extracted with the same feature schema version |
| `font_hash` | *(deferred)* | Always `None` — needs `/Resources/Font` tree-walking, not yet implemented |

`PdfFingerprint.matching_fields(other)` returns which of these are non-`None`
and equal — the minimal "fingerprints should be comparable" hook from the
platform's design brief. A full similarity/scoring engine is out of scope;
that's Rule Engine/ML territory.

## Shared refactor: `cos_type_name()`

`domain/pdf/objects.py` gained a public `cos_type_name(value) -> str` (and
`COS_TYPE_NAMES` listing every possible name), factored out of what was
previously a private table inside `objects_extractor.py`. Both Module 2's
`objects.type_histogram` feature and this module's `structure_hash` need the
exact same COS-type classification — sharing it means they can never
silently drift apart on what a given type is called.
