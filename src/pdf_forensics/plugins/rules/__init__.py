"""Forensic rules.

Explicitly wired (not auto-registered): `default_rules()`/
`default_entity_aware_rules()` below are the one place to touch when adding,
removing, or swapping a rule.

Deferred to later modules — none of these are attempted here, and none are
represented by placeholder findings: font inconsistencies, XMP mismatch,
linearization anomalies.

`default_entity_aware_rules()` is separate from `default_rules()` because
its rules need an `EntityIdentificationReport`, not just a `FeatureSet` —
see `application/rule_engine/entity_aware_ports.py`.
"""

from __future__ import annotations

from pdf_forensics.application.rule_engine.entity_aware_ports import EntityAwareRulePlugin
from pdf_forensics.application.rule_engine.ports import RulePlugin
from pdf_forensics.plugins.rules.broken_xref_rule import BrokenXrefRule
from pdf_forensics.plugins.rules.creation_after_mod_date_rule import CreationAfterModDateRule
from pdf_forensics.plugins.rules.duplicate_object_ids_rule import DuplicateObjectIdsRule
from pdf_forensics.plugins.rules.entity_template_mismatch_rule import EntityTemplateMismatchRule
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


def default_entity_aware_rules() -> tuple[EntityAwareRulePlugin, ...]:
    return (EntityTemplateMismatchRule(),)


__all__ = ["default_rules", "default_entity_aware_rules"]
