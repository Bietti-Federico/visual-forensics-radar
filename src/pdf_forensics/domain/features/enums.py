"""Closed vocabularies for `Feature.value_type` and `Feature.category`.

Both are stable string-valued enums, same contract as `AnomalyCode`
(`domain/pdf/anomalies.py`): existing members are never renamed or removed,
only added to, so a feature name and category already written to a report or
feature store stays meaningful forever.
"""

from __future__ import annotations

from enum import Enum


class FeatureType(Enum):
    """The runtime type of a `Feature.value`.

    Kept as an explicit enum rather than relying on `type(value)` so it stays
    serializable metadata a feature store can persist alongside the value —
    the same reasoning behind `PdfNumber.is_integer` being an explicit flag.
    """

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"
    LIST = "list"
    DICT = "dict"


class FeatureCategory(Enum):
    """The feature categories this module extracts.

    Only categories genuinely computable from the Module 1 `PdfDocument`
    model today — fonts, images, XMP semantics, signatures and linearization
    are explicitly deferred (see `plugins/features/__init__.py`), not
    represented here with empty/placeholder members.
    """

    GENERAL = "general"
    METADATA = "metadata"
    TRAILER = "trailer"
    CATALOG = "catalog"
    XREF = "xref"
    INCREMENTAL_UPDATES = "incremental_updates"
    OBJECTS = "objects"
    STREAMS = "streams"
    SECURITY = "security"
    STATISTICS = "statistics"
