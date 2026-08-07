"""Converts domain report objects into plain JSON-serializable dicts for the API."""

from __future__ import annotations

from typing import Any

from pdf_forensics.application.document_scoring.score_document_use_case import (
    DocumentScoringResult,
)


def serialize_scoring_result(result: DocumentScoringResult) -> dict[str, Any]:
    return {
        "fingerprint": result.fingerprint.to_dict(),
        "entity_predictions": [
            {
                "classifier_id": prediction.classifier_id,
                "predicted_entity": prediction.predicted_entity,
                "confidence": prediction.confidence,
                "probabilities": dict(prediction.probabilities),
            }
            for prediction in result.entity_report
        ],
        "signatures": [
            {
                "field_name": signature.field_name,
                "digest_intact": signature.digest_intact,
                "cryptographically_valid": signature.cryptographically_valid,
                "coverage": signature.coverage.value,
                "signer_subject": signature.signer_subject,
                "signing_time": (
                    signature.signing_time.isoformat() if signature.signing_time else None
                ),
            }
            for signature in result.signature_report
        ],
        "risk_score": result.risk_report.risk_score,
        "components": [
            {"name": component.name, "score": component.score, "weight": component.weight}
            for component in result.risk_report.components
        ],
        "reasons": list(result.explanation.reasons),
        "top_features": [
            {
                "model_id": feature.model_id,
                "feature_name": feature.feature_name,
                "shap_value": feature.shap_value,
            }
            for feature in result.explanation.top_features
        ],
    }
