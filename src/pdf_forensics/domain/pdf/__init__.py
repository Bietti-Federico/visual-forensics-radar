"""Domain model for a parsed PDF: the COS object system, the document aggregate and anomalies."""

from pdf_forensics.domain.pdf.anomalies import (
    AnomalyCode,
    AnomalyCollector,
    AnomalySeverity,
    StructuralAnomaly,
)
from pdf_forensics.domain.pdf.document import (
    PdfDocument,
    Revision,
    XrefEntry,
    XrefEntryType,
)
from pdf_forensics.domain.pdf.errors import (
    NotAPdfError,
    PdfForensicsError,
    UnrecoverableStructureError,
)
from pdf_forensics.domain.pdf.objects import (
    PdfArray,
    PdfBoolean,
    PdfDictionary,
    PdfHexString,
    PdfLiteralString,
    PdfName,
    PdfNull,
    PdfNumber,
    PdfObject,
    PdfReference,
    PdfStream,
    PdfValue,
)

__all__ = [
    "AnomalyCode",
    "AnomalyCollector",
    "AnomalySeverity",
    "StructuralAnomaly",
    "PdfDocument",
    "Revision",
    "XrefEntry",
    "XrefEntryType",
    "NotAPdfError",
    "PdfForensicsError",
    "UnrecoverableStructureError",
    "PdfArray",
    "PdfBoolean",
    "PdfDictionary",
    "PdfHexString",
    "PdfLiteralString",
    "PdfName",
    "PdfNull",
    "PdfNumber",
    "PdfObject",
    "PdfReference",
    "PdfStream",
    "PdfValue",
]
