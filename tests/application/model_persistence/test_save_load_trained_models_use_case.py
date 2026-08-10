from pathlib import Path

import pytest

from pdf_forensics.application.model_persistence.entity_model_bundle import EntityModelBundle
from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)
from pdf_forensics.application.model_persistence.save_trained_models_use_case import (
    SaveTrainedModelsUseCase,
)
from pdf_forensics.domain.entity_invariants.learned_invariant import LearnedInvariant
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

    invariants = (LearnedInvariant("catalog.has_acroform", True),)
    bundle = EntityModelBundle(
        detectors=detectors,
        models=models,
        invariants=invariants,
        genuine_count=15,
        confirmed_fraud_count=15,
        ml_ensemble_ready=True,
    )
    output_path = tmp_path / "model_store.joblib"
    SaveTrainedModelsUseCase().execute(entity_classifiers, {"ENTITY_A": bundle}, output_path)

    loaded_entity_classifiers, per_entity = LoadTrainedModelsUseCase().execute(output_path)

    assert {c.classifier_id for c in loaded_entity_classifiers} == {
        c.classifier_id for c in entity_classifiers
    }
    assert set(per_entity) == {"ENTITY_A"}
    loaded_bundle = per_entity["ENTITY_A"]
    assert loaded_bundle.genuine_count == 15
    assert loaded_bundle.confirmed_fraud_count == 15
    assert loaded_bundle.ml_ensemble_ready is True
    assert loaded_bundle.invariants == invariants
    assert {d.detector_id for d in loaded_bundle.detectors} == {d.detector_id for d in detectors}
    assert {m.model_id for m in loaded_bundle.models} == {m.model_id for m in models}

    for original in detectors:
        loaded_by_id = next(
            d for d in loaded_bundle.detectors if d.detector_id == original.detector_id
        )
        before = original.score(OUTLIER_VECTOR)
        after = loaded_by_id.score(OUTLIER_VECTOR)
        assert after.score == pytest.approx(before.score)
        assert after.is_anomaly == before.is_anomaly

    for original in models:
        loaded_by_id = next(m for m in loaded_bundle.models if m.model_id == original.model_id)
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


def test_entity_with_no_fitted_models_round_trips_as_empty_tuples(tmp_path: Path) -> None:
    entity_classifiers = default_entity_classifiers()
    bundle = EntityModelBundle(
        detectors=(),
        models=(),
        invariants=(),
        genuine_count=1,
        confirmed_fraud_count=0,
        ml_ensemble_ready=False,
    )
    output_path = tmp_path / "model_store.joblib"
    SaveTrainedModelsUseCase().execute(entity_classifiers, {"NEW_ENTITY": bundle}, output_path)

    _, per_entity = LoadTrainedModelsUseCase().execute(output_path)

    loaded_bundle = per_entity["NEW_ENTITY"]
    assert loaded_bundle.detectors == ()
    assert loaded_bundle.models == ()
    assert loaded_bundle.invariants == ()
    assert loaded_bundle.ml_ensemble_ready is False


def test_load_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LoadTrainedModelsUseCase().execute(tmp_path / "does_not_exist.joblib")


def test_load_raises_for_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad_store.joblib"
    save_bundle(path, {"schema_version": "0.0.1", "entity_classifiers": {}, "per_entity": {}})

    with pytest.raises(ValueError, match="schema_version"):
        LoadTrainedModelsUseCase().execute(path)


def test_schema_version_constant_is_stable() -> None:
    assert MODEL_STORE_SCHEMA_VERSION == "3.0.0"
