from pathlib import Path

from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)
from pdf_forensics.application.model_training.retrain_models_use_case import (
    RetrainModelsUseCase,
)
from tests.fixtures.pdf_builder import PdfBuilder


def _pdf_bytes(variant: int) -> bytes:
    builder = PdfBuilder()
    off1 = builder.add_object(1, 0, "<< /Type /Catalog /Pages 2 0 R >>")
    off2 = builder.add_object(2, 0, f"<< /Type /Pages /Kids [] /Count {variant} >>")
    builder.add_classic_xref_and_trailer(
        [(0, 0, 65535, "f"), (1, off1, 0, "n"), (2, off2, 0, "n")], size=3, root_ref="1 0 R"
    )
    return builder.build()


def _write_docs(directory: Path, count: int, start: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"doc_{start + i}.pdf").write_bytes(_pdf_bytes(start + i))


def test_entity_below_anomaly_threshold_gets_no_detectors(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=1)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    assert summary.total_entries == 1
    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.genuine_count == 1
    assert entity_summary.anomaly_detection_fitted is False
    assert entity_summary.ml_ensemble_ready is False

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert per_entity["ANSES"].detectors == ()


def test_entity_above_anomaly_threshold_gets_detectors(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=3)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.anomaly_detection_fitted is True

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert len(per_entity["ANSES"].detectors) > 0


def test_entity_with_zero_genuine_docs_gets_no_invariants(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    # Confirmed-fraud only, no genuine documents at all for this entity.
    _write_docs(corpus_dir / "confirmed_fraud" / "ANSES", count=1)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.invariant_count == 0

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert per_entity["ANSES"].invariants == ()


def test_single_genuine_doc_is_enough_for_invariants(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=1)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.invariant_count > 0

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert len(per_entity["ANSES"].invariants) > 0


def test_new_entity_gets_invariants_mined_with_no_code_changes(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "BRAND_NEW_ENTITY", count=4)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    entity_summary = next(s for s in summary.per_entity if s.entity == "BRAND_NEW_ENTITY")
    assert entity_summary.invariant_count > 0


def test_ml_ensemble_activates_only_when_both_thresholds_met(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=6)
    _write_docs(corpus_dir / "confirmed_fraud" / "ANSES", count=2, start=100)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)
    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.ml_ensemble_ready is False  # only 2 fraud, needs 3

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert per_entity["ANSES"].models == ()


def test_ml_ensemble_activates_once_threshold_crossed(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=6)
    _write_docs(corpus_dir / "confirmed_fraud" / "ANSES", count=3, start=100)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)
    entity_summary = next(s for s in summary.per_entity if s.entity == "ANSES")
    assert entity_summary.ml_ensemble_ready is True

    _, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert len(per_entity["ANSES"].models) > 0


def test_new_entity_folder_is_picked_up_automatically(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_docs(corpus_dir / "genuine" / "ANSES", count=2)
    _write_docs(corpus_dir / "genuine" / "BRAND_NEW_ENTITY", count=2, start=50)

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    entities = {s.entity for s in summary.per_entity}
    assert entities == {"ANSES", "BRAND_NEW_ENTITY"}

    entity_classifiers, _ = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert len(entity_classifiers) > 0


def test_empty_corpus_produces_empty_bundle_without_raising(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    assert summary.total_entries == 0
    assert summary.per_entity == ()

    entity_classifiers, per_entity = LoadTrainedModelsUseCase().execute(tmp_path / "model.joblib")
    assert entity_classifiers == ()
    assert per_entity == {}


def test_skipped_files_are_reported_not_fatal(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    genuine_dir = corpus_dir / "genuine" / "ANSES"
    _write_docs(genuine_dir, count=1)
    (genuine_dir / "corrupt.pdf").write_bytes(b"not a pdf")

    summary = RetrainModelsUseCase(tmp_path / "model.joblib").execute(corpus_dir)

    assert summary.total_entries == 1
    assert summary.skipped_count == 1
