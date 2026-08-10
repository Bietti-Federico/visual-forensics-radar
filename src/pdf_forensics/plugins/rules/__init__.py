"""Forensic rules.

Explicitly wired (not auto-registered): `default_rules()` below is the one
place to touch when adding, removing, or swapping a rule.

Deferred to later modules — none of these are attempted here, and none are
represented by placeholder findings: font inconsistencies, XMP mismatch,
linearization anomalies.

There used to be a second, entity-aware category here
(`default_entity_aware_rules()`/`EntityAwareRulePlugin`) for a single rule,
`EntityTemplateMismatchRule`, that checked a document against a hand-written
per-entity dict of structural invariants. That rule is retired: the same
signal is now mined automatically per entity from the training corpus (see
`application/entity_invariants/fit_entity_invariants_use_case.py`) rather
than hand-maintained here, and — since it needs fitted per-entity state, not
just a `FeatureSet` — it's invoked directly from `ScoreDocumentUseCase`
alongside Anomaly Detection/ML Ensemble, not through this plugin registry.
"""

from __future__ import annotations

from pdf_forensics.application.rule_engine.ports import RulePlugin
from pdf_forensics.plugins.rules.broken_xref_rule import BrokenXrefRule
from pdf_forensics.plugins.rules.creation_after_mod_date_rule import CreationAfterModDateRule
from pdf_forensics.plugins.rules.duplicate_object_ids_rule import DuplicateObjectIdsRule
from pdf_forensics.plugins.rules.missing_info_dictionary_rule import MissingInfoDictionaryRule
from pdf_forensics.plugins.rules.pdf_version_mismatch_rule import PdfVersionMismatchRule
from pdf_forensics.plugins.rules.trailer_missing_root_rule import TrailerMissingRootRule
from pdf_forensics.plugins.rules.unexpected_incremental_update_rule import (
    UnexpectedIncrementalUpdateRule,
)


def default_rules() -> tuple[RulePlugin, ...]:
    return (
        CreationAfterModDateRule(),
        PdfVersionMismatchRule(),
        MissingInfoDictionaryRule(),
        DuplicateObjectIdsRule(),
        BrokenXrefRule(),
        UnexpectedIncrementalUpdateRule(),
        TrailerMissingRootRule(),
    )


__all__ = ["default_rules"]
