import pytest

from pdf_forensics.plugins.anomaly_detection.one_class_svm_detector import OneClassSVMDetector
from tests.fixtures.anomaly_detection_fixtures import INLIER_VECTOR, NORMAL_VECTORS, OUTLIER_VECTOR


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="one_class_svm"):
        OneClassSVMDetector().score(NORMAL_VECTORS[0])


def test_outlier_scores_higher_than_inlier() -> None:
    detector = OneClassSVMDetector()
    detector.fit(NORMAL_VECTORS)

    inlier_score = detector.score(INLIER_VECTOR)
    outlier_score = detector.score(OUTLIER_VECTOR)

    assert outlier_score.score > inlier_score.score
    assert outlier_score.is_anomaly is True


def test_detector_id() -> None:
    assert OneClassSVMDetector().detector_id == "one_class_svm"
