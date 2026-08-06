from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore


def test_fields() -> None:
    score = AnomalyScore(detector_id="isolation_forest", score=1.5, is_anomaly=True)
    assert score.detector_id == "isolation_forest"
    assert score.score == 1.5
    assert score.is_anomaly is True
