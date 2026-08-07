"""
Scores one PDF through the full pipeline using a previously trained model
bundle (see scripts/retrain.py) — no retraining, instant scoring. Thin CLI
wrapper around `ScoreDocumentUseCase`, the same use case the API's
`/verify` endpoint calls.

Usage:
    poetry run python scripts/score_document.py <model_store_path> <target_pdf>
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdf_forensics.application.document_scoring.score_document_use_case import (
    ScoreDocumentUseCase,
)
from pdf_forensics.application.model_persistence.load_trained_models_use_case import (
    LoadTrainedModelsUseCase,
)


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    model_store_path = Path(sys.argv[1])
    target_pdf = Path(sys.argv[2])

    entity_classifiers, per_entity = LoadTrainedModelsUseCase().execute(model_store_path)
    result = ScoreDocumentUseCase(entity_classifiers, per_entity).execute(target_pdf.read_bytes())

    print(f"Fingerprint: {result.fingerprint.to_dict()}")
    print()
    print("Entity identification:")
    for prediction in result.entity_report:
        print(
            f"  [{prediction.classifier_id}] {prediction.predicted_entity} "
            f"(confidence={prediction.confidence:.0%}) — {prediction.probabilities}"
        )
    print()
    print("Signature verification:")
    if not result.signature_report:
        print("  No embedded signature found.")
    for sig_result in result.signature_report:
        print(
            f"  [{sig_result.field_name}] intact={sig_result.digest_intact} "
            f"valid={sig_result.cryptographically_valid} coverage={sig_result.coverage.value} "
            f"signer={sig_result.signer_subject!r} signed_at={sig_result.signing_time}"
        )
    print()
    print(f"Risk Score: {result.risk_report.risk_score}/100")
    print("Components:")
    for component in result.risk_report.components:
        print(f"  {component.name}: score={component.score:.3f} weight={component.weight}")
    print("Reasons:")
    for reason in result.explanation.reasons:
        print(f"  - {reason}")
    print("Top SHAP features:")
    for feature in result.explanation.top_features:
        print(f"  [{feature.model_id}] {feature.feature_name}: {feature.shap_value:+.4f}")


if __name__ == "__main__":
    main()
