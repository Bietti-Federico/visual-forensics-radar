"""Domain model for anomaly detection: unsupervised scores over a document's feature vector."""

from pdf_forensics.domain.anomaly_detection.anomaly_detection_report import AnomalyDetectionReport
from pdf_forensics.domain.anomaly_detection.anomaly_score import AnomalyScore

__all__ = ["AnomalyDetectionReport", "AnomalyScore"]
