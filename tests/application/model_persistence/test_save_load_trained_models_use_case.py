from pathlib import Path

import pytest

from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)
from pdf_forensics.application.model_persistence.save_trained_models_use_case import (
    SaveTrainedModelsUseCase,
)
from pdf_forensics.infrastructure.model_persistence.model_store import (
    MODEL_STORE_SCHEMA_VERSION,
    save_bundle,
)
from pdf_forensics.plugins.anomaly_detection import default_detectors
from pdf_forensics.plugins.entity_identification import default_entity_classifiers
from pdf_forensics.plugins.ml_ensemble import default_models
from tests.fixtures.anomaly_detection_fixtures import INLIER_VECTOR, NORMAL_VECTORS, OUTLIER_VECTOR


def _labeled_batch() -> tuple[list[dict[str, float]], list[bool]]:
    originals = NORMAL_VECTORS[:15]
    manipulated = [{"a": 100.0 + i, "b": 100.0 + i} for i in range(15)]
    feature_vectors = originals + manipulated
    labels = [True] * len(originals) + [False] * len(manipulated)
    return feature_vectors, labels


def _entity_labeled_batch() -> tuple[list[dict[str, float]], list[str]]:
    entity_a = [{"a": v, "b": v} for v in (0.0, 0.1, 0.2, 0.3)]
    entity_b = [{"a": 100.0 + v, "b": 100.0 + v} for v in (0.0, 0.1, 0.2, 0.3)]
    feature_vectors = entity_a + entity_b
    entity_labels = ["ENTITY_A"] * len(entity_a) + ["ENTITY_B"] * len(entity_b)
    return feature_vectors, entity_labels


def test_round_trip_predictions_match(tmp_path: Path) -> None:
    detectors = default_detectors()
    for detector in detectors:
        detector.fit(NORMAL_VECTORS)

    models = default_models()
    feature_vectors, labels = _labeled_batch()
    for model in models:
        model.fit(feature_vectors, labels)

    entity_classifiers = default_entity_classifiers()
    entity_feature_vectors, entity_labels = _entity_labeled_batch()
    for classifier in entity_classifiers:
        classifier.fit(entity_feature_vectors, entity_labels)

    output_path = tmp_path / "model_store.joblib"
    SaveTrainedModelsUseCase().execute(detectors, models, output_path, entity_classifiers)

    loaded_detectors, loaded_models, loaded_entity_classifiers = LoadTrainedModelsUseCase().execute(
        output_path
    )

    assert {d.detector_id for d in loaded_detectors} == {d.detector_id for d in detectors}
    assert {m.model_id for m in loaded_models} == {m.model_id for m in models}
    assert {c.classifier_id for c in loaded_entity_classifiers} == {
        c.classifier_id for c in entity_classifiers
    }

    for original in detectors:
        loaded_by_id = next(d for d in loaded_detectors if d.detector_id == original.detector_id)
        before = original.score(OUTLIER_VECTOR)
        after = loaded_by_id.score(OUTLIER_VECTOR)
        assert after.score == pytest.approx(before.score)
        assert after.is_anomaly == before.is_anomaly

    for original in models:
        loaded_by_id = next(m for m in loaded_models if m.model_id == original.model_id)
        before = original.predict(INLIER_VECTOR)
        after = loaded_by_id.predict(INLIER_VECTOR)
        assert after.probability == pytest.approx(before.probability)
        assert after.predicted_label == before.predicted_label

    for original in entity_classifiers:
        loaded_by_id = next(
            c for c in loaded_entity_classifiers if c.classifier_id == original.classifier_id
        )
        before = original.predict(entity_feature_vectors[0])
        after = loaded_by_id.predict(entity_feature_vectors[0])
        assert after.confidence == pytest.approx(before.confidence)
        assert after.predicted_entity == before.predicted_entity


def test_load_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LoadTrainedModelsUseCase().execute(tmp_path / "does_not_exist.joblib")


def test_load_raises_for_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad_store.joblib"
    save_bundle(path, {"schema_version": "0.0.1", "detectors": {}, "models": {}})

    with pytest.raises(ValueError, match="schema_version"):
        LoadTrainedModelsUseCase().execute(path)


def test_schema_version_constant_is_stable() -> None:
    assert MODEL_STORE_SCHEMA_VERSION == "1.0.0"
