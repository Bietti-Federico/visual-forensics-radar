import pytest

from pdf_forensics.plugins.anomaly_detection.local_outlier_factor_detector import (
    LocalOutlierFactorDetector,
)
from tests.fixtures.anomaly_detection_fixtures import INLIER_VECTOR, NORMAL_VECTORS, OUTLIER_VECTOR


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="local_outlier_factor"):
        LocalOutlierFactorDetector().score(NORMAL_VECTORS[0])


def test_outlier_scores_higher_than_inlier() -> None:
    detector = LocalOutlierFactorDetector()
    detector.fit(NORMAL_VECTORS)

    inlier_score = detector.score(INLIER_VECTOR)
    outlier_score = detector.score(OUTLIER_VECTOR)

    assert outlier_score.score > inlier_score.score
    assert outlier_score.is_anomaly is True


def test_n_neighbors_capped_for_small_fit_batches() -> None:
    # Default n_neighbors=20 would raise on a 3-sample fit batch; this must not.
    detector = LocalOutlierFactorDetector()
    small_batch = NORMAL_VECTORS[:3]
    detector.fit(small_batch)
    result = detector.score(small_batch[0])
    assert result.detector_id == "local_outlier_factor"


def test_detector_id() -> None:
    assert LocalOutlierFactorDetector().detector_id == "local_outlier_factor"
