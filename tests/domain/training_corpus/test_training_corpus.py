from pathlib import Path

from pdf_forensics.domain.training_corpus.training_corpus import TrainingCorpus
from pdf_forensics.domain.training_corpus.training_corpus_entry import TrainingCorpusEntry
from tests.fixtures.feature_helpers import build_feature_set


def _entry(entity: str, is_genuine: bool, name: str) -> TrainingCorpusEntry:
    return TrainingCorpusEntry(
        entity=entity, is_genuine=is_genuine, path=Path(name), features=build_feature_set({})
    )


def test_by_entity_groups_entries() -> None:
    entries = [
        _entry("ANSES", True, "a.pdf"),
        _entry("ANSES", False, "b.pdf"),
        _entry("LA_RIOJA", True, "c.pdf"),
    ]
    corpus = TrainingCorpus(entries=entries)

    grouped = corpus.by_entity()

    assert set(grouped) == {"ANSES", "LA_RIOJA"}
    assert len(grouped["ANSES"]) == 2
    assert len(grouped["LA_RIOJA"]) == 1


def test_empty_corpus_groups_to_empty_dict() -> None:
    assert TrainingCorpus().by_entity() == {}
