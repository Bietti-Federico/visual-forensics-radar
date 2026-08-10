"""Thin adapter around pyHanko: reads a PDF's embedded PKCS#7 signatures and
translates pyHanko's own result types into this platform's domain types.

Deliberately operates on raw PDF bytes via pyHanko's own reader, not this
platform's Module 1 `PdfDocument` — signature validation is a distinct,
well-defined problem pyHanko already solves correctly (CMS/PKCS#7 parsing,
digest recomputation over `/ByteRange`), the same reasoning
`pdf-forensics-benchmark` uses qpdf/exiftool as external infrastructure
adapters instead of reimplementing their checks.

Trust-chain validation (`status.trusted`) is intentionally never surfaced —
see `domain/signature_verification/signature_verification_result.py`'s
module docstring for why.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Coroutine, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pyhanko.pdf_utils.reader import PdfFileReader
from pyhanko.sign.validation.pdf_embedded import async_validate_pdf_signature
from pyhanko.sign.validation.status import SignatureCoverageLevel

from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)
from pdf_forensics.domain.signature_verification.signature_verification_result import (
    SignatureVerificationResult,
)

_COVERAGE_MAP = {
    SignatureCoverageLevel.ENTIRE_FILE: SignatureCoverage.ENTIRE_FILE,
    SignatureCoverageLevel.ENTIRE_REVISION: SignatureCoverage.PARTIAL,
    SignatureCoverageLevel.CONTIGUOUS_BLOCK_FROM_START: SignatureCoverage.PARTIAL,
    SignatureCoverageLevel.UNCLEAR: SignatureCoverage.UNCLEAR,
}

_T = TypeVar("_T")


def _run_coros_sync(coros: Sequence[Coroutine[Any, Any, _T]]) -> list[_T | BaseException]:
    """pyHanko's signature validation is async internally; this adapter is
    called synchronously from both plain scripts and FastAPI's async
    `/verify` handler. `asyncio.run()` raises if a loop is already running
    (as it is inside a FastAPI request) — that used to be silently
    swallowed by a broad `except Exception`, dropping every signature found
    on documents verified through the API. Running every signature's
    coroutine together through one dedicated thread/event loop (instead of
    spinning up a new thread and loop per signature) sidesteps the
    already-running loop and avoids that per-signature setup/teardown cost
    on multiply-signed documents. `return_exceptions=True` keeps one
    signature's validation failure from cancelling the others."""

    async def _gather() -> list[_T | BaseException]:
        # `asyncio.gather(*coros)` must be constructed inside the coroutine
        # that `asyncio.run()` actually drives — calling it beforehand binds
        # it to whatever event loop happens to be current at that moment
        # (or none), which `asyncio.run()` then rejects as "not a coroutine".
        return await asyncio.gather(*coros, return_exceptions=True)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_gather())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, _gather()).result()


def read_signature_verification_report(pdf_bytes: bytes) -> SignatureVerificationReport:
    """
    Never raises: a document pyHanko can't even parse as a PDF (this
    platform's own Module 1 parser is far more tolerant of malformed input)
    is reported as having no signatures, matching Module 1's own philosophy
    of anomalies over exceptions — a signature check that can't run isn't
    itself forensic evidence of anything.
    """
    try:
        reader = PdfFileReader(io.BytesIO(pdf_bytes))
        embedded_signatures = list(reader.embedded_signatures)
    except Exception:
        return SignatureVerificationReport()

    statuses = _run_coros_sync(
        [async_validate_pdf_signature(signature) for signature in embedded_signatures]
    )

    results = []
    for embedded_signature, status in zip(embedded_signatures, statuses, strict=True):
        if isinstance(status, BaseException):
            continue
        results.append(
            SignatureVerificationResult(
                field_name=embedded_signature.field_name,
                digest_intact=status.intact,
                cryptographically_valid=status.valid,
                coverage=(
                    _COVERAGE_MAP[status.coverage]
                    if status.coverage is not None
                    else SignatureCoverage.UNCLEAR
                ),
                signer_subject=(
                    status.signing_cert.subject.human_friendly if status.signing_cert else None
                ),
                signing_time=status.signer_reported_dt,
            )
        )
    return SignatureVerificationReport(results=results)
