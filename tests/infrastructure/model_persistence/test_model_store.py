from pathlib import Path

import pytest

from pdf_forensics.infrastructure.model_persistence.model_store import (
    MODEL_STORE_SCHEMA_VERSION,
    load_bundle,
    save_bundle,
)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "model_store.joblib"
    save_bundle(path, {"foo": "bar"})

    loaded = load_bundle(path)

    assert loaded["foo"] == "bar"
    assert loaded["schema_version"] == MODEL_STORE_SCHEMA_VERSION


def test_save_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    path = tmp_path / "model_store.joblib"
    save_bundle(path, {"foo": "bar"})

    assert list(tmp_path.iterdir()) == [path]


def test_save_does_not_corrupt_existing_bundle_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "model_store.joblib"
    save_bundle(path, {"foo": "original"})

    class Unpicklable:
        def __reduce__(self) -> tuple[object, ...]:
            raise RuntimeError("cannot pickle this")

    with pytest.raises(RuntimeError):
        save_bundle(path, {"foo": Unpicklable()})

    # The atomic-write temp file must be cleaned up, and the original,
    # still-valid bundle must be untouched by the failed write.
    assert list(tmp_path.iterdir()) == [path]
    assert load_bundle(path)["foo"] == "original"


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle("/nonexistent/path/model_store.joblib")


def test_load_rejects_mismatched_schema_version(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "model_store.joblib"
    joblib.dump({"schema_version": "0.0.1"}, path)

    with pytest.raises(ValueError, match="schema_version"):
        load_bundle(path)
