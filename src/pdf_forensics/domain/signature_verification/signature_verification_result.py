"""One embedded signature field's verification result.

`trusted` is deliberately excluded — validating a certificate against a
trust root needs the actual issuing authority's root certificate configured
(e.g. Argentina's ONTI root for AFIP/ANSES-issued certificates), which this
platform doesn't ship. Reporting "not trusted" without that configured would
read as a red flag on every single genuine signature, which is worse than
not reporting it at all — see `docs/signature-verification.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage


@dataclass(frozen=True, slots=True)
class SignatureVerificationResult:
    field_name: str
    #: Whether the cryptographic message digest matches the bytes actually
    #: covered by `/ByteRange` — `False` means the signed content changed
    #: since signing (the strongest possible tamper signal this platform can
    #: produce).
    digest_intact: bool
    #: Whether the PKCS#7/CMS signature itself is a well-formed, valid
    #: signature over that digest for the claimed signer's public key —
    #: independent of whether that key is trusted.
    cryptographically_valid: bool
    coverage: SignatureCoverage
    signer_subject: str | None
    signing_time: datetime | None
