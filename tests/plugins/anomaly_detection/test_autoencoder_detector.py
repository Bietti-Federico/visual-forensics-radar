import pytest

from pdf_forensics.plugins.anomaly_detection.autoencoder_detector import (
    AutoencoderAnomalyDetector,
)
from tests.fixtures.anomaly_detection_fixtures import INLIER_VECTOR, NORMAL_VECTORS, OUTLIER_VECTOR


def _fast_detector() -> AutoencoderAnomalyDetector:
    # Tiny architecture and few epochs: this is a correctness test, not a
    # quality-of-fit benchmark, so speed matters more than convergence.
    return AutoencoderAnomalyDetector(hidden_dim=4, bottleneck_dim=2, epochs=50)


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="autoencoder"):
        _fast_detector().score(NORMAL_VECTORS[0])


def test_outlier_reconstructs_worse_than_inlier() -> None:
    detector = _fast_detector()
    detector.fit(NORMAL_VECTORS)

    inlier_score = detector.score(INLIER_VECTOR)
    outlier_score = detector.score(OUTLIER_VECTOR)

    assert outlier_score.score > inlier_score.score
    assert outlier_score.is_anomaly is True
    assert inlier_score.is_anomaly is False


def test_detector_id() -> None:
    assert AutoencoderAnomalyDetector().detector_id == "autoencoder"


def test_fit_is_deterministic_given_fixed_seed() -> None:
    detector_a = AutoencoderAnomalyDetector(
        hidden_dim=4, bottleneck_dim=2, epochs=20, random_seed=7
    )
    detector_b = AutoencoderAnomalyDetector(
        hidden_dim=4, bottleneck_dim=2, epochs=20, random_seed=7
    )
    detector_a.fit(NORMAL_VECTORS)
    detector_b.fit(NORMAL_VECTORS)

    assert detector_a.score(NORMAL_VECTORS[0]).score == detector_b.score(NORMAL_VECTORS[0]).score
