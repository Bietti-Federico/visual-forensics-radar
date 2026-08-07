from pathlib import Path

from pdf_forensics.infrastructure.training_corpus.filesystem_training_corpus_reader import (
    read_training_corpus_files,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")


def test_reads_genuine_and_confirmed_fraud_files_grouped_by_entity(tmp_path: Path) -> None:
    _touch(tmp_path / "genuine" / "ANSES" / "a.pdf")
    _touch(tmp_path / "genuine" / "ANSES" / "b.pdf")
    _touch(tmp_path / "genuine" / "LA_RIOJA" / "c.pdf")
    _touch(tmp_path / "confirmed_fraud" / "ANSES" / "d.pdf")

    refs = read_training_corpus_files(tmp_path)

    assert len(refs) == 4
    genuine_anses = [r for r in refs if r.entity == "ANSES" and r.is_genuine]
    assert {r.path.name for r in genuine_anses} == {"a.pdf", "b.pdf"}
    fraud_anses = [r for r in refs if r.entity == "ANSES" and not r.is_genuine]
    assert [r.path.name for r in fraud_anses] == ["d.pdf"]
    genuine_la_rioja = [r for r in refs if r.entity == "LA_RIOJA"]
    assert len(genuine_la_rioja) == 1 and genuine_la_rioja[0].is_genuine


def test_ignores_non_pdf_files(tmp_path: Path) -> None:
    _touch(tmp_path / "genuine" / "ANSES" / "a.pdf")
    (tmp_path / "genuine" / "ANSES" / "readme.txt").write_text("not a pdf")

    refs = read_training_corpus_files(tmp_path)

    assert len(refs) == 1
    assert refs[0].path.name == "a.pdf"


def test_missing_directories_do_not_raise(tmp_path: Path) -> None:
    assert read_training_corpus_files(tmp_path) == []


def test_new_entity_folder_requires_no_code_change(tmp_path: Path) -> None:
    _touch(tmp_path / "genuine" / "BRAND_NEW_ENTITY" / "x.pdf")

    refs = read_training_corpus_files(tmp_path)

    assert len(refs) == 1
    assert refs[0].entity == "BRAND_NEW_ENTITY"
