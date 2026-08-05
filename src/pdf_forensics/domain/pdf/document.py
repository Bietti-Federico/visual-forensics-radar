"""The parsed-PDF aggregate: revisions, the resolved object graph, and anomalies.

A PDF file is not one snapshot — every incremental update appends a new xref
section and trailer on top of the previous ones (ISO 32000-1 §7.5.6), and the
*sequence* of those updates is itself forensic evidence (an unexpected extra
save, a resized xref, an object silently redefined). `PdfDocument` keeps that
history explicit via `revisions` rather than collapsing straight to a single
merged view, while still offering the merged view (`trailer`, `resolve`) for
callers that just want "the current state of the file."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pdf_forensics.domain.pdf.anomalies import StructuralAnomaly
from pdf_forensics.domain.pdf.objects import PdfDictionary, PdfObject, PdfReference, PdfValue


class XrefEntryType(Enum):
    """The three kinds of cross-reference entry (ISO 32000-1 §7.5.4, §7.5.7)."""

    FREE = "free"
    IN_USE = "in_use"
    COMPRESSED = "compressed"


@dataclass(frozen=True, slots=True)
class XrefEntry:
    """One row of a cross-reference section (table- or stream-based) for one object.

    `offset_or_stream_obj_num` is a byte offset from the file start when
    `entry_type` is `IN_USE`, or the object number of the containing object
    stream when `entry_type` is `COMPRESSED`; it is meaningless (0) when
    `entry_type` is `FREE`. `index_in_stream` is only meaningful for `COMPRESSED`
    entries — see ISO 32000-1 Table 18, field types 1 and 2.
    """

    obj_num: int
    generation: int
    entry_type: XrefEntryType
    offset_or_stream_obj_num: int
    index_in_stream: int | None = None


@dataclass(frozen=True, slots=True)
class Revision:
    """One xref section + trailer, corresponding to one save of the file.

    The *first* revision parsed is the most recent save (the one `startxref`
    points to); earlier saves are reached by following `/Prev` and appear later
    in `PdfDocument.revisions`.
    """

    trailer: PdfDictionary
    xref_entries: dict[int, XrefEntry]
    xref_offset: int
    is_xref_stream: bool


@dataclass(slots=True)
class PdfDocument:
    """Aggregate root: everything this platform's parser could determine about one PDF.

    `is_encrypted`/`encryption_dict` are recorded here, and left otherwise
    unused, so a future decryption module can populate decoded string/stream
    content without changing this type's shape. Likewise `objects` stores
    `PdfObject`s exactly as parsed — a stream's `raw_data` may still be
    filter-encoded if its filter isn't implemented by this module (see
    `PdfStream.filter_names` and `AnomalyCode.UNSUPPORTED_FILTER`).
    """

    revisions: list[Revision]
    objects: dict[tuple[int, int], PdfObject]
    anomalies: list[StructuralAnomaly] = field(default_factory=list)
    is_encrypted: bool = False
    encryption_dict: PdfDictionary | None = None
    raw_size: int = 0
    pdf_version: str | None = None

    @property
    def latest_revision(self) -> Revision:
        return self.revisions[0]

    @property
    def trailer(self) -> PdfDictionary:
        """The trailer as of the latest revision, with older revisions' keys filling gaps.

        Per ISO 32000-1 §7.5.5, a trailer only needs to repeat keys that changed;
        `/Root` or `/Info` from an earlier save still apply if a later trailer
        omits them. Later (more recent) revisions win on key conflicts.
        """
        merged: dict[str, PdfValue] = {}
        for revision in reversed(self.revisions):
            merged.update(revision.trailer.entries)
        return PdfDictionary(entries=merged)

    def resolve(self, ref: PdfReference) -> PdfObject | None:
        """Look up an indirect object by reference, or `None` if never defined."""
        return self.objects.get((ref.obj_num, ref.generation))
