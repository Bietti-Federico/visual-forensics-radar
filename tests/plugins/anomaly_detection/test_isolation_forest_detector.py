import pytest

from pdf_forensics.plugins.anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector,
)
from tests.fixtures.anomaly_detection_fixtures import INLIER_VECTOR, NORMAL_VECTORS, OUTLIER_VECTOR


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="isolation_forest"):
        IsolationForestDetector().score(NORMAL_VECTORS[0])


def test_outlier_scores_higher_than_inlier() -> None:
    detector = IsolationForestDetector()
    detector.fit(NORMAL_VECTORS)

    inlier_score = detector.score(INLIER_VECTOR)
    outlier_score = detector.score(OUTLIER_VECTOR)

    assert outlier_score.score > inlier_score.score
    assert outlier_score.is_anomaly is True
    assert inlier_score.is_anomaly is False


def test_detector_id() -> None:
    assert IsolationForestDetector().detector_id == "isolation_forest"
