from __future__ import annotations

import asyncio
import datetime
from io import BytesIO
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign import fields, signers
from pyhanko.sign.fields import SigFieldSpec
from pyhanko.sign.signers.pdf_signer import PdfSignatureMetadata
from reportlab.pdfgen import canvas

from pdf_forensics.application.signature_verification.verify_signatures_use_case import (
    VerifySignaturesUseCase,
)
from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)

_SIGNED_TEXT = b"Hello signed world"


def _make_signer(tmp_path: Path) -> signers.SimpleSigner:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Signer")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )
    key_path = tmp_path / "key.pem"
    cert_path = tmp_path / "cert.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return signers.SimpleSigner.load(str(key_path), str(cert_path))


def _unsigned_pdf_bytes() -> bytes:
    buf = BytesIO()
    # pageCompression=0: keeps the content stream as plain, searchable bytes
    # instead of FlateDecode-compressed — the tampering test below needs to
    # locate and flip a byte inside the actual signed text.
    c = canvas.Canvas(buf, pageCompression=0)
    c.drawString(100, 700, _SIGNED_TEXT.decode())
    c.save()
    return buf.getvalue()


def _signed_pdf_bytes(tmp_path: Path) -> bytes:
    signer = _make_signer(tmp_path)
    writer = IncrementalPdfFileWriter(BytesIO(_unsigned_pdf_bytes()))
    fields.append_signature_field(writer, SigFieldSpec(sig_field_name="Signature1"))
    output = signers.sign_pdf(writer, PdfSignatureMetadata(field_name="Signature1"), signer=signer)
    return output.getvalue()


def test_valid_signature_is_intact_and_covers_entire_file(tmp_path: Path) -> None:
    report = VerifySignaturesUseCase().execute(_signed_pdf_bytes(tmp_path))

    assert len(report) == 1
    result = report.results[0]
    assert result.field_name == "Signature1"
    assert result.digest_intact is True
    assert result.cryptographically_valid is True
    assert result.coverage is SignatureCoverage.ENTIRE_FILE
    assert result.signer_subject is not None
    assert "Test Signer" in result.signer_subject
    assert result.signing_time is not None


def test_tampering_after_signing_breaks_digest(tmp_path: Path) -> None:
    signed = bytearray(_signed_pdf_bytes(tmp_path))
    # Flip a byte inside the visible page content stream — guaranteed to
    # fall within the signed /ByteRange rather than accidentally landing
    # inside the /Contents placeholder itself.
    target = signed.index(_SIGNED_TEXT)
    signed[target] = ord("h")  # 'H' -> 'h'

    report = VerifySignaturesUseCase().execute(bytes(signed))

    assert len(report) == 1
    assert report.results[0].digest_intact is False


def test_unsigned_pdf_has_no_signature_results() -> None:
    report = VerifySignaturesUseCase().execute(_unsigned_pdf_bytes())

    assert len(report) == 0


def test_garbage_bytes_do_not_raise() -> None:
    report = VerifySignaturesUseCase().execute(b"this is not a pdf at all")

    assert len(report) == 0


def test_valid_signature_found_when_called_from_a_running_event_loop(tmp_path: Path) -> None:
    """pyHanko validates signatures with its own internal `asyncio.run()`,
    which raises if a loop is already running (as it is inside a FastAPI
    request handler). Regression test for that: the signature must still be
    found and reported when this use case executes on the same thread as a
    running event loop, not silently dropped."""
    signed = _signed_pdf_bytes(tmp_path)

    async def _verify_from_within_a_loop() -> SignatureVerificationReport:
        return VerifySignaturesUseCase().execute(signed)

    report = asyncio.run(_verify_from_within_a_loop())

    assert len(report) == 1
    assert report.results[0].digest_intact is True
