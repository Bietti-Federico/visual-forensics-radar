"""The complete set of signature verification results for one document."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from pdf_forensics.domain.signature_verification.signature_verification_result import (
    SignatureVerificationResult,
)


@dataclass(slots=True)
class SignatureVerificationReport:
    results: list[SignatureVerificationResult] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[SignatureVerificationResult]:
        return iter(self.results)
