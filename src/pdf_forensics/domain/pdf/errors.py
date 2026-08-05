"""Exceptions reserved for input this platform cannot extract any evidence from.

Everything short of total unparseability is modeled as a `StructuralAnomaly`
(see `anomalies.py`), not an exception — a forensics tool that raises on the first
malformed byte would be useless against exactly the files it's meant to examine.
"""

from __future__ import annotations


class PdfForensicsError(Exception):
    """Base class for exceptions raised by this platform."""


class NotAPdfError(PdfForensicsError):
    """No `%PDF-` header was found anywhere in the scanned prefix of the file."""


class UnrecoverableStructureError(PdfForensicsError):
    """No object could be located by any recovery strategy.

    Raised only when neither a classic xref, an xref stream, nor a brute-force
    `N G obj` byte scan finds anything — a `%PDF-` header alone is not evidence
    of a document to analyze.
    """
