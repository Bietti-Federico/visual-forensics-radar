from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore


def test_by_detector_returns_matching_score_or_none() -> None:
    a = AnomalyScore(detector_id="isolation_forest", score=1.0, is_anomaly=False)
    b = AnomalyScore(detector_id="autoencoder", score=2.0, is_anomaly=True)
    report = AnomalyDetectionReport(scores=[a, b])

    assert report.by_detector("isolation_forest") is a
    assert report.by_detector("autoencoder") is b
    assert report.by_detector("missing") is None


def test_len_and_iter() -> None:
    scores = [
        AnomalyScore(detector_id="a", score=1.0, is_anomaly=False),
        AnomalyScore(detector_id="b", score=2.0, is_anomaly=True),
    ]
    report = AnomalyDetectionReport(scores=scores)
    assert len(report) == 2
    assert list(report) == scores


def test_empty_report_defaults() -> None:
    report = AnomalyDetectionReport()
    assert len(report) == 0
    assert report.by_detector("anything") is None
