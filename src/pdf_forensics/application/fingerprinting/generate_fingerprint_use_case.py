"""Turns a `PdfDocument` + its `FeatureSet` into a `PdfFingerprint`.

Only traverses the already-built `PdfDocument` graph (dict/reference lookups,
the same operations Module 2's extractors perform) and consumes a `FeatureSet`
for exactly one field, `feature_hash` — it never re-tokenizes raw bytes.

Deliberately does NOT look up `feature_set.by_name("metadata.producer")` etc.
for `generator`/`producer`/`pdf_version`: that would couple this module to
Module 2's feature-name strings by convention only, with no compiler or test
to catch a rename. Instead it re-resolves `/Info` directly from `document` (a
small, self-contained helper intentionally similar to, but not shared with,
`metadata_extractor.py` — decoupling is worth the minor duplication here) and
reads `document.pdf_version` directly.
"""

from __future__ import annotations

import hashlib
import json

from pdf_forensics.domain.features.enums import FeatureCategory
from pdf_forensics.domain.features.feature_set import FeatureSet
from pdf_forensics.domain.fingerprint.fingerprint import PdfFingerprint
from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.domain.pdf.objects import (
    PdfDictionary,
    PdfHexString,
    PdfLiteralString,
    PdfReference,
    cos_type_name,
)

HASH_ALGORITHM = "sha256"
FINGERPRINT_SCHEMA_VERSION = "1.0.0"

_UNRESOLVED_PAGE_TREE_SENTINEL = "UNRESOLVED"


class GenerateFingerprintUseCase:
    def execute(self, document: PdfDocument, feature_set: FeatureSet) -> PdfFingerprint:
        info = self._resolve_info_dict(document)
        return PdfFingerprint(
            generator=self._string_value(info, "Creator") if info is not None else None,
            producer=self._string_value(info, "Producer") if info is not None else None,
            pdf_version=document.pdf_version,
            xref_hash=self._compute_xref_hash(document),
            structure_hash=self._compute_structure_hash(document),
            metadata_hash=self._compute_metadata_hash(feature_set),
            font_hash=None,
            page_tree_hash=self._compute_page_tree_hash(document),
            feature_hash=self._compute_feature_hash(feature_set),
            algorithm=HASH_ALGORITHM,
            schema_version=FINGERPRINT_SCHEMA_VERSION,
        )

    def _compute_xref_hash(self, document: PdfDocument) -> str:
        entries = document.latest_revision.xref_entries
        lines = [
            f"{obj_num}|{entry.generation}|{entry.entry_type.value}|{entry.offset_or_stream_obj_num}"
            for obj_num, entry in sorted(entries.items())
        ]
        return self._hash_lines(lines)

    def _compute_structure_hash(self, document: PdfDocument) -> str:
        lines = [
            f"{obj_num}|{generation}|{cos_type_name(obj.value)}"
            for (obj_num, generation), obj in sorted(document.objects.items())
        ]
        return self._hash_lines(lines)

    def _compute_metadata_hash(self, feature_set: FeatureSet) -> str:
        lines = [
            f"{feature.name}={feature.value!r}"
            for feature in sorted(
                feature_set.by_category(FeatureCategory.METADATA), key=lambda f: f.name
            )
        ]
        return self._hash_lines(lines)

    def _compute_page_tree_hash(self, document: PdfDocument) -> str:
        catalog = self._resolve_dict(document, document.trailer.get_ref("Root"))
        pages = self._resolve_dict(document, catalog.get_ref("Pages")) if catalog else None
        if pages is None:
            return self._hash_lines([_UNRESOLVED_PAGE_TREE_SENTINEL])

        type_name = pages.get_name("Type") or ""
        count = pages.get_int("Count")
        kids = pages.get_array("Kids")
        kid_refs = [
            f"{item.obj_num} {item.generation} R"
            for item in (kids.items if kids else ())
            if isinstance(item, PdfReference)
        ]
        return self._hash_lines([f"Type={type_name}|Count={count}|Kids=[{','.join(kid_refs)}]"])

    def _compute_feature_hash(self, feature_set: FeatureSet) -> str:
        canonical = json.dumps(feature_set.to_dict(), sort_keys=True, default=self._json_fallback)
        return hashlib.new(HASH_ALGORITHM, canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _json_fallback(value: object) -> object:
        if isinstance(value, tuple):
            return list(value)
        return str(value)

    def _hash_lines(self, lines: list[str]) -> str:
        canonical = "\n".join(lines)
        return hashlib.new(HASH_ALGORITHM, canonical.encode("utf-8")).hexdigest()

    def _resolve_info_dict(self, document: PdfDocument) -> PdfDictionary | None:
        info_ref = document.trailer.get_ref("Info")
        if info_ref is None:
            return None
        return self._resolve_dict(document, info_ref)

    def _resolve_dict(
        self, document: PdfDocument, ref: PdfReference | None
    ) -> PdfDictionary | None:
        if ref is None:
            return None
        obj = document.resolve(ref)
        if obj is None or not isinstance(obj.value, PdfDictionary):
            return None
        return obj.value

    def _string_value(self, dictionary: PdfDictionary, key: str) -> str | None:
        value = dictionary.get(key)
        if isinstance(value, PdfLiteralString | PdfHexString):
            return value.decode_text()
        return None
