# Signature Verification

Covers `src/pdf_forensics/domain/signature_verification/`,
`src/pdf_forensics/application/signature_verification/`, and
`src/pdf_forensics/infrastructure/signature_verification/`.

## What this is

Validates a PDF's embedded PKCS#7/CMS digital signatures (if any): does the
cryptographic message digest still match the signed bytes (`digest_intact`),
is the signature itself well-formed and mathematically valid
(`cryptographically_valid`), and does it cover the entire file or leave
room for content appended after signing (`coverage`).

Real-world validation: the actual ANSES receipts this platform trained on
carry a genuine PKCS#7 detached signature (`/Filter /Adobe.PPKLite`,
`/SubFilter /adbe.pkcs7.detached`) behind their "Firmado Digitalmente"
badge. Verifying it end-to-end (`intact=True`, `valid=True`,
`coverage=ENTIRE_FILE`) is what motivated this module — Entity
Identification's `catalog.has_acroform` invariant only checks that a
signature *field* exists, not that whatever's in it is genuine.

## Why an external library, unlike Module 1's own parser

PKCS#7/CMS parsing and cryptographic verification is deliberately **not**
hand-rolled from scratch, unlike Module 1's own from-scratch PDF object
parser. This is exactly the kind of security-sensitive parsing (ASN.1, RSA/
ECDSA signature verification, certificate handling) where reimplementing it
is the wrong call — [pyHanko](https://github.com/MatthiasValvekens/pyHanko)
is a well-tested, actively maintained library built specifically for this,
the same reasoning `pdf-forensics-benchmark` uses qpdf/exiftool as external
infrastructure adapters instead of reimplementing their checks.

This module therefore operates on raw PDF bytes via pyHanko's own reader,
not this platform's Module 1 `PdfDocument` — a deliberate, documented
exception to "everything goes through our own parser."

## Trust-chain validation is explicitly out of scope

pyHanko can also validate a signing certificate against a trust root, but
doing so needs that root configured (e.g. Argentina's ONTI root CA for
AFIP/ANSES-issued certificates), which this platform doesn't ship.
`status.trusted` is never surfaced: reporting "not trusted" without a
configured root would flag every single genuine signature red, which is
worse than not reporting it. `SignatureVerificationResult` intentionally has
no `trusted` field — see its own docstring.

## Risk Report integration: `signature_integrity`

`domain/risk/risk_weights.py` gives this component weight `0.10`. Scoring,
worst-finding-wins across every signature found:

| Finding | Score | Why |
|---|---|---|
| No signature present | `0.0` | Not itself suspicious — most documents this platform handles have none. `entity_template_mismatch` (Rule Engine) already covers "this entity's real documents always have one and this doesn't." |
| `digest_intact=False` | `1.0` | The strongest possible tamper signal: content changed since signing. |
| `coverage != ENTIRE_FILE` | `0.7` | Content exists that was never signed (e.g. added after signing). |
| `cryptographically_valid=False` | `0.9` | Malformed/forged signature blob. |
| Fully valid signature | `0.0` | Genuine, unmodified, entirely-covered signature. |

## Usage

```python
from pdf_forensics.application.signature_verification.verify_signatures_use_case import (
    VerifySignaturesUseCase,
)

signature_report = VerifySignaturesUseCase().execute(pdf_bytes)
for result in signature_report:
    print(result.field_name, result.digest_intact, result.cryptographically_valid, result.coverage)
```
