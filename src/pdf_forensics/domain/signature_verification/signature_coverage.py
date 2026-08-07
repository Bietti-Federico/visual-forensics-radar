"""How much of the file one embedded signature's `/ByteRange` actually covers.

A closed vocabulary, same contract as `AnomalyCode`/`FeatureType`: existing
members are never renamed or removed, only added to.
"""

from __future__ import annotations

from enum import Enum


class SignatureCoverage(Enum):
    #: `/ByteRange` covers every byte of the file except the signature's own
    #: placeholder — no content could have been appended after signing.
    ENTIRE_FILE = "entire_file"
    #: `/ByteRange` covers less than the whole file — bytes exist that were
    #: never signed (e.g. a later incremental update added content after
    #: this signature, or after an unrelated visual annotation).
    PARTIAL = "partial"
    #: The underlying library couldn't determine coverage confidently.
    UNCLEAR = "unclear"
