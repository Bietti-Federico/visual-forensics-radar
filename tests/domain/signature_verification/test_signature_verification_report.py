from pdf_forensics.domain.signature_verification.signature_coverage import SignatureCoverage
from pdf_forensics.domain.signature_verification.signature_verification_report import (
    SignatureVerificationReport,
)
from pdf_forensics.domain.signature_verification.signature_verification_result import (
    SignatureVerificationResult,
)


def _result(field_name: str) -> SignatureVerificationResult:
    return SignatureVerificationResult(
        field_name=field_name,
        digest_intact=True,
        cryptographically_valid=True,
        coverage=SignatureCoverage.ENTIRE_FILE,
        signer_subject=None,
        signing_time=None,
    )


def test_len_and_iteration() -> None:
    results = [_result("Signature1"), _result("Signature2")]
    report = SignatureVerificationReport(results=results)

    assert len(report) == 2
    assert list(report) == results


def test_empty_report_has_no_results() -> None:
    report = SignatureVerificationReport()

    assert len(report) == 0
    assert list(report) == []
