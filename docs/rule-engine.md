# Module 4: Rule Engine

Covers `src/pdf_forensics/domain/rules/`, `src/pdf_forensics/application/rule_engine/`,
`src/pdf_forensics/plugins/rules/`, and the small addition `src/pdf_forensics/domain/pdf/pdf_date.py`.

## What this is, and isn't

Declarative, independent rules that each look at an already-extracted
`FeatureSet` (Module 2) and return `severity`, `confidence`, `explanation`,
`references` — or nothing, if the rule doesn't find anything to flag. Not a
risk score (a later module combines findings + anomalies + ML output into
one); not generator identification (a separate future ML classifier).

## Rules couple to feature names on purpose

Module 3 (Fingerprint) deliberately avoided looking up features by name
string, to stay decoupled from Module 2's naming. Rules are the opposite
case on purpose: a rule **is** a declarative statement about a specific named
feature (`metadata.has_info_dict is False`, ...) — that coupling is the point
of this abstraction, not something to engineer around.

## The 7 rules and why each severity was chosen

| Rule | Trigger | Severity | Why |
|---|---|---|---|
| `creation_after_mod_date` | `/CreationDate` after `/ModDate`, UTC-normalized | WARNING | Both dates present and comparable — a document appearing modified before it was created |
| `pdf_version_mismatch` | header version ≠ `/Root/Version` | INFO | A real, spec-sanctioned override (ISO 32000-1 §7.5.2) — not inherently suspicious |
| `missing_info_dictionary` | no `/Info` dict | WARNING | |
| `duplicate_object_ids` | `objects.duplicate_object_id_count > 0` | WARNING | |
| `broken_xref` | xref-related anomalies present | CRITICAL/WARNING | Mirrors the severity Module 1 already assigned the same anomaly code — never reported at two different severities |
| `unexpected_incremental_update` | `revision_count` above threshold (default 2) | WARNING | Ports the legacy tool's baseline: 2 saves normal, 3+ worth a look |
| `trailer_missing_root` | no `/Root` in the trailer | WARNING | Matches Module 1's `TRAILER_MISSING_ROOT` severity |

**Explicitly deferred**, same pattern as prior modules: font inconsistencies,
XMP mismatch, linearization anomalies (all need capabilities not yet built).

## An 8th rule, added once Entity Identification existed: `entity_template_mismatch`

`plugins/rules/entity_template_mismatch_rule.py`, wired via a *separate*
`default_entity_aware_rules()` (not `default_rules()`) because it needs an
`EntityIdentificationReport`, not just a `FeatureSet` — see
`application/rule_engine/entity_aware_ports.py` for why that's a distinct,
narrower `Protocol` rather than a breaking change to `RulePlugin` itself.
Flags a document confidently identified as a known entity (ANSES / La Rioja
/ Jujuy) whose structure contradicts that entity's known invariant
(`catalog.has_acroform`) — see `docs/entity-identification.md`.

Callers run both use cases and merge their `RuleEvaluationReport`s (plain
list concatenation — the type has no notion of "which use case produced
this finding") before handing the result to Explainability/Risk Report,
which need no changes to consume entity-aware findings.

## A rule never guesses

Every rule returns `None`, not a low-confidence finding, when the features it
needs are missing or unparseable (e.g. `creation_after_mod_date` needs *both*
dates to parse successfully). `None` means "nothing to say" — covering both
"no issue" and "can't tell" without conflating them into a false signal.

## Plugin architecture: explicit wiring, same as Module 2

`RulePlugin` (`application/rule_engine/ports.py`) is a `Protocol`, not an ABC.
`plugins/rules/__init__.py`'s `default_rules()` is the one explicit tuple to
edit when adding/removing/swapping a rule — no `__init_subclass__`
auto-registration, for the same reasons documented in
`docs/feature-extraction.md`.
