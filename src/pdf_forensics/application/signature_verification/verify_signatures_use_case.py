"""Application boundary for verifying a PDF's embedded digital signatures, if any.

Takes raw PDF bytes (not a `FeatureSet`) — see
`infrastructure/signature_verification/pyhanko_adapter.py`'s module
docstring for why signature validation deliberately bypasses Module 1's own
parser. No plugin/Protocol layer here (unlike Modules 4-6, 9): there is
exactly one correct way to validate a PKCS#7 signature, this isn't a
pluggable-strategy problem.
"""

from __future__ import annotations

from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)
from pdf_forensics.infrastructure.signature_verification.pyhanko_adapter import (
    read_signature_verification_report,
)


class VerifySignaturesUseCase:
    def execute(self, pdf_bytes: bytes) -> SignatureVerificationReport:
        return read_signature_verification_report(pdf_bytes)
