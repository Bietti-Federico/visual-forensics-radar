import pytest

from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.domain.pdf.errors import NotAPdfError
from pdf_forensics.domain.pdf.objects import PdfReference
from tests.fixtures.pdf_builder import PdfBuilder


def test_execute_parses_a_minimal_document() -> None:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")],
        size=3,
        root_ref="1 0 R",
    )

    document = ParsePdfUseCase().execute(builder.build())

    assert document.trailer.get_ref("Root") == PdfReference(1, 0)


def test_execute_raises_domain_error_on_non_pdf_bytes() -> None:
    with pytest.raises(NotAPdfError):
        ParsePdfUseCase().execute(b"definitely not a pdf")
