"""A PDF's fingerprint: a comparable, hash-based summary of its structure and metadata.

Not a risk score and not generator identification (both later modules) —
just a deterministic, canonical set of hashes plus a couple of raw metadata
strings, cheap to compute and cheap to compare between two documents.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Field names participating in `matching_fields()` — every hash this module
#: computes. `generator`/`producer`/`pdf_version` are raw copied strings, not
#: hashes, so they're excluded from a *fingerprint* comparison on purpose.
_HASH_FIELD_NAMES = (
    "xref_hash",
    "structure_hash",
    "metadata_hash",
    "font_hash",
    "page_tree_hash",
    "feature_hash",
)


@dataclass(frozen=True, slots=True)
class PdfFingerprint:
    generator: str | None
    producer: str | None
    pdf_version: str | None
    xref_hash: str
    structure_hash: str
    metadata_hash: str
    #: Always `None` this iteration — computing it needs `/Resources/Font`
    #: tree-walking, explicitly deferred (see Module 2's deferred list). The
    #: field exists now so a later module can populate it without changing
    #: this type's shape.
    font_hash: str | None
    page_tree_hash: str
    feature_hash: str
    algorithm: str
    schema_version: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "generator": self.generator,
            "producer": self.producer,
            "pdf_version": self.pdf_version,
            "xref_hash": self.xref_hash,
            "structure_hash": self.structure_hash,
            "metadata_hash": self.metadata_hash,
            "font_hash": self.font_hash,
            "page_tree_hash": self.page_tree_hash,
            "feature_hash": self.feature_hash,
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
        }

    def matching_fields(self, other: PdfFingerprint) -> frozenset[str]:
        """Which hash fields are non-`None` and equal between `self` and `other`.

        `font_hash` can never appear here today since it's always `None` — a
        `None == None` "match" would claim two documents are similar on a
        dimension neither fingerprint actually computed.
        """
        return frozenset(
            name
            for name in _HASH_FIELD_NAMES
            if getattr(self, name) is not None and getattr(self, name) == getattr(other, name)
        )
