"""Structural anomalies: malformed PDF structure as first-class forensic data.

A general-purpose PDF reader treats a broken xref offset, a duplicate object id or
a mismatched `/Length` as a bug to route around silently so it can still render the
page. This platform exists to do the opposite: those exact deviations from a
well-formed file are the signal. The parser therefore never aborts on recoverable
malformation — it records a `StructuralAnomaly` and keeps going, so the final
`PdfDocument` carries both the best-effort reconstruction of the content *and* the
complete list of ways the file failed to conform to spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pdf_forensics.domain.pdf.objects import PdfReference


class AnomalySeverity(Enum):
    """How much a given anomaly, on its own, should move a forensic risk estimate."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyCode(Enum):
    """Catalog of structural deviations this parser can detect.

    Each member's value is the stable string used in serialized reports — do not
    rename existing members' values, only add new ones, so historical reports
    remain comparable.
    """

    XREF_OFFSET_MISMATCH = "xref_offset_mismatch"
    DUPLICATE_OBJECT_ID = "duplicate_object_id"
    TRAILER_MISSING_ROOT = "trailer_missing_root"
    STREAM_LENGTH_MISMATCH = "stream_length_mismatch"
    XREF_UNPARSEABLE_FALLBACK_USED = "xref_unparseable_fallback_used"
    OBJSTM_INDEX_OUT_OF_RANGE = "objstm_index_out_of_range"
    UNSUPPORTED_FILTER = "unsupported_filter"
    BROKEN_PREV_CHAIN = "broken_prev_chain"
    FILTER_DECODE_FAILED = "filter_decode_failed"


@dataclass(frozen=True, slots=True)
class StructuralAnomaly:
    """One detected deviation from well-formed PDF structure."""

    code: AnomalyCode
    severity: AnomalySeverity
    message: str
    object_ref: PdfReference | None = None
    byte_offset: int | None = None


@dataclass(slots=True)
class AnomalyCollector:
    """Mutable accumulator threaded through the parser call chain.

    A plain list would work too, but a dedicated collector gives every parsing
    component one narrow interface (`record`) instead of each needing to know the
    final `PdfDocument.anomalies` list shape, and makes it trivial to add
    deduplication or rate-limiting later without touching call sites.
    """

    anomalies: list[StructuralAnomaly] = field(default_factory=list)

    def record(
        self,
        code: AnomalyCode,
        severity: AnomalySeverity,
        message: str,
        *,
        object_ref: PdfReference | None = None,
        byte_offset: int | None = None,
    ) -> None:
        self.anomalies.append(
            StructuralAnomaly(
                code=code,
                severity=severity,
                message=message,
                object_ref=object_ref,
                byte_offset=byte_offset,
            )
        )

    def to_list(self) -> list[StructuralAnomaly]:
        return list(self.anomalies)
