from pathlib import Path

from pdf_forensics.application.feature_extraction.extract_features_use_case import (
    FeatureExtractionUseCase,
)
from pdf_forensics.application.pdf_analysis.parse_pdf_use_case import ParsePdfUseCase
from pdf_forensics.application.training_corpus.build_training_corpus_use_case import (
    BuildTrainingCorpusUseCase,
)
from pdf_forensics.plugins.features import default_feature_extractors
from tests.fixtures.pdf_builder import PdfBuilder


def _valid_pdf_bytes() -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, "<< /Type /Pages /Kids [] /Count 0 >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder.build()


def _use_case() -> BuildTrainingCorpusUseCase:
    return BuildTrainingCorpusUseCase(
        parse_pdf_use_case=ParsePdfUseCase(),
        feature_extraction_use_case=FeatureExtractionUseCase(default_feature_extractors()),
    )


def test_reads_genuine_and_confirmed_fraud_entries(tmp_path: Path) -> None:
    genuine_dir = tmp_path / "genuine" / "ANSES"
    genuine_dir.mkdir(parents=True)
    (genuine_dir / "a.pdf").write_bytes(_valid_pdf_bytes())

    fraud_dir = tmp_path / "confirmed_fraud" / "ANSES"
    fraud_dir.mkdir(parents=True)
    (fraud_dir / "b.pdf").write_bytes(_valid_pdf_bytes())

    corpus = _use_case().execute(tmp_path)

    assert len(corpus.entries) == 2
    assert corpus.skipped == []
    genuine_entries = [e for e in corpus.entries if e.is_genuine]
    fraud_entries = [e for e in corpus.entries if not e.is_genuine]
    assert len(genuine_entries) == 1
    assert len(fraud_entries) == 1
    assert genuine_entries[0].entity == "ANSES"
    assert genuine_entries[0].features.by_name("general.object_count").value == 2


def test_corrupt_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    genuine_dir = tmp_path / "genuine" / "ANSES"
    genuine_dir.mkdir(parents=True)
    (genuine_dir / "good.pdf").write_bytes(_valid_pdf_bytes())
    (genuine_dir / "bad.pdf").write_bytes(b"this is not a pdf at all")

    corpus = _use_case().execute(tmp_path)

    assert len(corpus.entries) == 1
    assert corpus.entries[0].path.name == "good.pdf"
    assert len(corpus.skipped) == 1
    assert corpus.skipped[0].path.name == "bad.pdf"
    assert corpus.skipped[0].entity == "ANSES"
