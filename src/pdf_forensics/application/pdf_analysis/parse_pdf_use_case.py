"""Application boundary for parsing a PDF, per Clean Architecture's dependency rule.

Deliberately thin: its only reason to exist separately from
`PdfDocumentParser.parse` directly is to be the one place future cross-cutting
concerns (logging, timing, telemetry) attach without infrastructure code needing
to know about them, and to be the stable import path other layers (API, CLI)
depend on instead of reaching into `infrastructure.parsing` directly.
"""

from __future__ import annotations

from pdf_forensics.domain.pdf.document import PdfDocument
from pdf_forensics.infrastructure.parsing.document_parser import PdfDocumentParser


class ParsePdfUseCase:
    def __init__(self, parser: PdfDocumentParser | None = None) -> None:
        self._parser = parser or PdfDocumentParser()

    def execute(self, pdf_bytes: bytes) -> PdfDocument:
        """Parse raw PDF bytes into a `PdfDocument`.

        Raises `NotAPdfError` or `UnrecoverableStructureError`
        (`pdf_forensics.domain.pdf.errors`) — no other exception is expected to
        escape a call to this method; anything else is a bug in the parser.
        """
        return self._parser.parse(pdf_bytes)
